import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def dwconv2d_fwd_kernel(
    x_ptr,         # *fptr: [N, C, H, W] contiguous
    w_ptr,         # *fptr: [C, K_H*K_W] flattened contiguous
    b_ptr,         # *fptr: [C] or dummy (unused if BIAS=0)
    y_ptr,         # *fptr: [N, C, H_OUT, W_OUT] contiguous
    N, C, H, W,    # int32
    H_OUT, W_OUT,  # int32
    BIAS: tl.constexpr,     # 0/1
    K_H: tl.constexpr, K_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DIL_H: tl.constexpr, DIL_W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program ids (over N*C and tiles of H_OUT*W_OUT) - do not change
    pid_nc = tl.program_id(0)
    pid_tile = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    total_hw = H_OUT * W_OUT
    mask_o = offs < total_hw

    oh = offs // W_OUT
    ow = offs % W_OUT

    # Base pointers
    base_x = (n * C + c) * H * W
    base_y = (n * C + c) * H_OUT * W_OUT

    # Precompute origins for input coordinates
    oh_base = oh * STRIDE_H - PAD_H
    ow_base = ow * STRIDE_W - PAD_W

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Per-channel weight base
    w_ch_base = c * (K_H * K_W)

    # Fast path: no padding and unit dilation -> all taps are guaranteed in-bounds
    if (PAD_H == 0) and (PAD_W == 0) and (DIL_H == 1) and (DIL_W == 1):
        for kh in tl.static_range(K_H):
            ih = oh_base + kh
            row_ptrs = x_ptr + base_x + ih * W + ow_base
            w_row_base = w_ptr + w_ch_base + kh * K_W
            for kw in tl.static_range(K_W):
                x_vals = tl.load(row_ptrs + kw, mask=mask_o, other=0.0)
                w_val = tl.load(w_row_base + kw)
                acc += x_vals.to(tl.float32) * w_val.to(tl.float32)
    else:
        # Generic path with full per-tap bounds checks
        for kh in tl.static_range(K_H):
            ih = oh_base + kh * DIL_H
            h_ok = (ih >= 0) & (ih < H)
            row_ptrs = x_ptr + base_x + ih * W + ow_base
            w_row_base = w_ptr + w_ch_base + kh * K_W
            for kw in tl.static_range(K_W):
                iw = ow_base + kw * DIL_W
                w_ok = (iw >= 0) & (iw < W)
                m = mask_o & h_ok & w_ok
                x_vals = tl.load(row_ptrs + kw * DIL_W, mask=m, other=0.0)
                w_val = tl.load(w_row_base + kw)
                acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    if BIAS:
        b = tl.load(b_ptr + c)
        acc += b.to(tl.float32)

    tl.store(y_ptr + base_y + offs, acc, mask=mask_o)


def _depthwise_conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> torch.Tensor:
    # Expect shapes:
    # x: [N, C, H, W], weight: [C, 1, K_H, K_W] (depthwise groups=C), bias: [C] or None
    assert x.ndim == 4 and weight.ndim == 4
    N, C, H, W = x.shape
    Cw, one, K_H, K_W = weight.shape
    assert Cw == C and one == 1, "Weight must be depthwise [C,1,K_H,K_W]"
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Output size (PyTorch conv2d formula)
    H_OUT = (H + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    W_OUT = (W + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1

    # Ensure contiguity
    x_c = x.contiguous()
    w_c = weight.contiguous().view(C, -1)  # [C, K_H*K_W]
    b_c = bias.contiguous() if bias is not None else None

    y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    # Tiling
    BLOCK_HW = 128
    grid = (N * C, triton.cdiv(H_OUT * W_OUT, BLOCK_HW))

    # Use a valid pointer for b_ptr even if BIAS=0 (it won't be accessed)
    dummy_bptr = x_c.view(-1)

    dwconv2d_fwd_kernel[grid](
        x_c, w_c.view(-1), (b_c if b_c is not None else dummy_bptr), y,
        N, C, H, W, H_OUT, W_OUT,
        BIAS=1 if b_c is not None else 0,
        K_H=K_H, K_W=K_W,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        DIL_H=dil_h, DIL_W=dil_w,
        BLOCK_HW=BLOCK_HW,
        num_warps=4, num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Match the original behavior: depthwise groups = in_channels; out_channels is ignored (must equal in_channels)
        self.conv2d = nn.Conv2d(
            in_channels, in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Fallback to PyTorch if not on CUDA
        if not x.is_cuda:
            return self.conv2d(x)
        # Triton implementation for CUDA tensors
        return _depthwise_conv2d_triton(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.conv2d.stride,
            self.conv2d.padding,
            self.conv2d.dilation,
        )


# Test code
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]