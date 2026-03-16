import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Triton acceleration
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@triton.jit
def _relu_copy_nchw_kernel(
    src_ptr, out_ptr,
    N, C, H, W,
    stride_ns, stride_cs, stride_hs, stride_ws,
    stride_no, stride_co, stride_ho, stride_wo,
    out_c_offset,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    w_idx = tl.arange(0, BLOCK_W)
    # Decompose pid into (n, c, h)
    h = pid % H
    tmp = pid // H
    c = tmp % C
    n = tmp // C

    # Guard against out-of-range program ids
    in_bounds_n = n < N
    in_bounds_c = c < C
    in_bounds_h = h < H
    valid_row = in_bounds_n & in_bounds_c & in_bounds_h

    # Compute base pointers
    src_base = n * stride_ns + c * stride_cs + h * stride_hs
    out_base = n * stride_no + (out_c_offset + c) * stride_co + h * stride_ho

    # Offsets along width
    mask_w = w_idx < W
    src_offsets = src_base + w_idx * stride_ws
    out_offsets = out_base + w_idx * stride_wo

    # Load -> ReLU -> Store
    vals = tl.load(src_ptr + src_offsets, mask=mask_w & valid_row, other=0.0, cache_modifier=".cg")
    vals = tl.maximum(vals, 0.0)
    tl.store(out_ptr + out_offsets, vals, mask=mask_w & valid_row, cache_modifier=".cg")


def _relu_concat_triton(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + concat along channel dim for two NCHW tensors using Triton.
    y1: (N, C1, H, W), y2: (N, C2, H, W)
    returns: (N, C1+C2, H, W)
    """
    assert y1.is_cuda and y2.is_cuda
    assert y1.shape[0] == y2.shape[0] and y1.shape[2] == y2.shape[2] and y1.shape[3] == y2.shape[3]
    N, C1, H, W = y1.shape
    C2 = y2.shape[1]
    out = torch.empty((N, C1 + C2, H, W), device=y1.device, dtype=y1.dtype)

    # Pick a good tile for width
    if W >= 256:
        BLOCK_W = 256
    elif W >= 128:
        BLOCK_W = 128
    elif W >= 64:
        BLOCK_W = 64
    else:
        BLOCK_W = 32

    # Launch for y1 -> out[:, :C1]
    grid1 = (N * C1 * H,)
    _relu_copy_nchw_kernel[grid1](
        y1, out,
        N, C1, H, W,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        0,
        BLOCK_W=BLOCK_W,
        num_warps=8 if W >= 128 else 4,
        num_stages=1,
    )

    # Launch for y2 -> out[:, C1:]
    grid2 = (N * C2 * H,)
    _relu_copy_nchw_kernel[grid2](
        y2, out,
        N, C2, H, W,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        C1,
        BLOCK_W=BLOCK_W,
        num_warps=8 if W >= 128 else 4,
        num_stages=1,
    )
    return out


def _should_use_triton(N: int, C1: int, C2: int, H: int, W: int) -> bool:
    # Heuristic: Triton is beneficial for wider rows or very large tiles.
    total_rows = N * (C1 + C2) * H
    return (W >= 128) or (W >= 64 and total_rows >= 8192)


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(FireModule, self).__init__()

        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        x = self.squeeze_activation(self.squeeze(x))
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)

        # Use Triton fused ReLU+concat when profitable
        if _TRITON_AVAILABLE and y1.is_cuda and y2.is_cuda and _should_use_triton(y1.shape[0], y1.shape[1], y2.shape[1], y1.shape[2], y1.shape[3]):
            return _relu_concat_triton(y1, y2)

        # PyTorch fallback: equivalent to applying ReLU to each branch then concat
        out = torch.cat([y1, y2], dim=1)
        return F.relu(out, inplace=True)


@triton.jit
def _adaptive_avg_pool2d_1x1_nchw_kernel(
    x_ptr,  # *f32 / *f16
    y_ptr,  # *f32 / *f16
    N, C,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c,
    H: tl.constexpr,  # spatial height (constexpr for unrolling)
    W: tl.constexpr,  # spatial width  (constexpr for vectorization)
):
    # One program handles one (n, c) pair.
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    # Base pointers for this (n, c)
    x_base = x_ptr + n * stride_n + c * stride_c
    y_ptr_scalar = y_ptr + n * out_stride_n + c * out_stride_c

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros((), tl.float32)

    # Vectorized reduction across W with compile-time tile size
    offs = tl.arange(0, 128)  # constexpr to satisfy Triton
    valid_nc = (n < N) & (c < C)

    for h in range(0, H):
        for w0 in range(0, W, 128):
            w_idx = w0 + offs
            mask = w_idx < W
            row_ptrs = x_base + h * stride_h + w_idx * stride_w
            vals = tl.load(row_ptrs, mask=mask & valid_nc, other=0.0).to(tl.float32)
            acc += tl.sum(vals, axis=0)

    mean = acc / float(H * W)
    tl.store(y_ptr_scalar, mean, mask=valid_nc)


def adaptive_avg_pool2d_1x1_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated AdaptiveAvgPool2d((1, 1)) for NCHW tensors.
    Falls back to PyTorch if Triton is not available or tensor is on CPU.
    """
    if (not _TRITON_AVAILABLE) or (not x.is_cuda) or (x.numel() == 0):
        return F.adaptive_avg_pool2d(x, (1, 1))

    # Ensure contiguous NCHW memory layout
    x = x.contiguous()
    N, C, H, W = x.shape
    # Output tensor retains input dtype
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, _, _ = y.stride()

    grid = (N * C,)
    # Launch kernel; accumulation is in fp32, but output dtype is preserved by y.
    _adaptive_avg_pool2d_1x1_nchw_kernel[grid](
        x, y,
        N, C,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c,
        H=H, W=W,
        num_warps=4, num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )

        # Keep semantics identical; pool in forward via Triton kernel.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        x = adaptive_avg_pool2d_1x1_triton(x)
        return torch.flatten(x, 1)


# Test code
batch_size = 1
input_channels = 3
height = 224
width = 224
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [num_classes]