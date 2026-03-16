import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 64,  'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 128, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 64,  'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 256, 'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'H', 'W', 'OC', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv2d_nchw_fp32_kernel(
    x_ptr,  # (N, C, H, W) float32
    w_ptr,  # (K, OC) float32 where K = C*KH*KW (packed layout)
    b_ptr,  # (OC,) float32
    y_ptr,  # (N, OC, H_OUT, W_OUT) float32
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_p = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)

    # Offsets along pixels (flattened N*H_OUT*W_OUT) and out-channels
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    total_p = N * H_OUT * W_OUT
    p_mask = p_offsets < total_p
    oc_mask = oc_offsets < OC

    # Decode flattened pixel indices into (n, y_out, x_out)
    WH_OUT = H_OUT * W_OUT
    n_idx = p_offsets // WH_OUT
    tmp = p_offsets % WH_OUT
    y_out = tmp // W_OUT
    x_out = tmp % W_OUT

    # Hints for better codegen
    tl.multiple_of(oc_offsets, BLOCK_OC)
    tl.max_contiguous(oc_offsets, BLOCK_OC)
    tl.multiple_of(p_offsets, BLOCK_P)
    tl.max_contiguous(p_offsets, BLOCK_P)

    # Initialize accumulator with bias
    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)
    bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Precompute constants to reduce index math
    HW = H * W
    KHW = KH * KW
    # n_idx * C * H * W
    n_idx_C = n_idx * C
    n_idx_C_HW = n_idx_C * HW  # [BLOCK_P]

    # Iterate over kernel elements and channels with static unrolling
    for ky in tl.static_range(0, KH):
        iy = y_out * STRIDE_H - PAD_H + ky * DIL_H  # [BLOCK_P]
        in_y_ok = (iy >= 0) & (iy < H)
        ky_base = ky * KW
        for kx in tl.static_range(0, KW):
            ix = x_out * STRIDE_W - PAD_W + kx * DIL_W  # [BLOCK_P]
            in_x_ok = (ix >= 0) & (ix < W)
            mask_p = p_mask & in_y_ok & in_x_ok

            # Base indices for current (ky, kx) across pixels
            base_hw = iy * W + ix                             # [BLOCK_P]
            x_base = n_idx_C_HW + base_hw                     # [BLOCK_P]

            # Precompute k-base for weights at (ky, kx)
            k_base_k = ky_base + kx  # in [0, KHW)

            # Loop over input channels; fully unrolled
            for ci in tl.static_range(0, C):
                # X index: ((n*C + ci) * H*W) + (iy*W + ix)
                x_index = x_base + ci * HW                    # [BLOCK_P]
                x_vals = tl.load(x_ptr + x_index, mask=mask_p, other=0.0).to(tl.float32)

                # Packed weights layout: [K, OC] with K fastest varying along OC
                # k_id = ci * (KH*KW) + ky*KW + kx
                k_id = ci * KHW + k_base_k
                w_index = k_id * OC + oc_offsets              # [BLOCK_OC]
                w_vals = tl.load(w_ptr + w_index, mask=oc_mask, other=0.0).to(tl.float32)

                # Outer product accumulate: (BLOCK_P, 1) * (1, BLOCK_OC)
                acc += x_vals[:, None] * w_vals[None, :]

    # Store results to Y
    y_index = ((n_idx[:, None] * OC + oc_offsets[None, :]) * H_OUT + y_out[:, None]) * W_OUT + x_out[:, None]
    store_mask = p_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + y_index, acc, mask=store_mask)


def _conv2d_triton_nchw(x: torch.Tensor,
                        weight: torch.Tensor,
                        bias: torch.Tensor | None,
                        stride: tuple[int, int],
                        padding: tuple[int, int],
                        dilation: tuple[int, int]) -> torch.Tensor:
    # Assumes x and weight are float32 and contiguous in NCHW / OIHW layouts.
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    N, C, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert C == Cw, "Input channels mismatch"
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    H_OUT = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_OUT = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    y = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=torch.float32)

    x_ = x.contiguous().to(torch.float32)
    # Pack weights to [K, OC] where K=C*KH*KW for coalesced OC loads
    K = C * KH * KW
    w_packed = weight.permute(1, 2, 3, 0).reshape(K, OC).contiguous().to(torch.float32)
    # Guarantee a valid bias pointer: zeros if module has no bias
    b_ = (bias.contiguous().to(torch.float32)
          if bias is not None
          else torch.zeros(OC, device=x.device, dtype=torch.float32))

    grid = lambda META: (
        triton.cdiv(N * H_OUT * W_OUT, META['BLOCK_P']),
        triton.cdiv(OC, META['BLOCK_OC']),
    )
    conv2d_nchw_fp32_kernel[grid](
        x_, w_packed, b_,
        y,
        N, C, H, W,
        OC, KH, KW,
        sh, sw,
        ph, pw,
        dh, dw,
        H_OUT, W_OUT,
    )
    return y.to(dtype=x.dtype) if x.dtype != torch.float32 else y


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width). 
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (tuple, optional): Padding applied to the input (top/bottom, left/right). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Fallback to PyTorch if not CUDA
        if not x.is_cuda:
            return self.conv2d(x)

        # Extract parameters
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        stride = self.conv2d.stride if isinstance(self.conv2d.stride, tuple) else (self.conv2d.stride, self.conv2d.stride)
        padding = self.conv2d.padding if isinstance(self.conv2d.padding, tuple) else (self.conv2d.padding, self.conv2d.padding)
        dilation = self.conv2d.dilation if isinstance(self.conv2d.dilation, tuple) else (self.conv2d.dilation, self.conv2d.dilation)

        return _conv2d_triton_nchw(x, weight, bias, stride, padding, dilation)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 256
height = 256
stride = 1
padding = (1, 2)  # Asymmetric padding
dilation = (2, 1)  # Asymmetric dilation

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]