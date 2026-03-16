import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Match grid's BLOCK_W=128 to ensure full coverage without changing grid logic
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
    ],
    key=['W_OUT'],
)
@triton.jit
def _dwconv2d_kernel(
    x_ptr,           # *const T, [N, C_in, H_in, W_in]
    w_ptr,           # *const T, [C_out, 1, K, K]
    b_ptr,           # *const T or nullptr if no bias, [C_out]
    y_ptr,           # *mut T,   [N, C_out, H_out, W_out]
    N: tl.constexpr,
    C_IN,
    C_OUT,
    H_IN,
    W_IN,
    H_OUT,
    W_OUT,
    STRIDE,
    PADDING,
    OCPG,           # out_channels per group (= out_channels // in_channels)
    K: tl.constexpr,               # kernel size (square)
    HAS_BIAS: tl.constexpr,        # compile-time flag
    BLOCK_W: tl.constexpr,         # tile size along W_out
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    # Decompose pid across (N, C_OUT)
    n = pid_nc // C_OUT
    oc = pid_nc % C_OUT
    ho = pid_h

    # Tile of output width this program computes
    w_start = pid_w * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W_OUT

    # Map output channel to its input channel (depthwise groups = in_channels)
    ic = oc // OCPG

    # Accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Precompute base scales for pointer arithmetic (int64 to avoid overflow)
    n = tl.full((), n, tl.int64)
    oc_i64 = tl.full((), oc, tl.int64)
    ic_i64 = tl.full((), ic, tl.int64)
    C_IN = tl.full((), C_IN, tl.int64)
    C_OUT = tl.full((), C_OUT, tl.int64)
    H_IN = tl.full((), H_IN, tl.int64)
    W_IN = tl.full((), W_IN, tl.int64)
    H_OUT = tl.full((), H_OUT, tl.int64)
    W_OUT_i64 = tl.full((), W_OUT, tl.int64)
    STRIDE = tl.full((), STRIDE, tl.int32)
    PADDING = tl.full((), PADDING, tl.int32)

    # Compute once per-row
    ho_i = ho * STRIDE

    # Base weight offset for this output channel
    w_oc_base = oc_i64 * (K * K)

    # Loop over kernel KxK
    for r in range(K):
        hi = ho_i - PADDING + r
        hi_in = (hi >= 0) & (hi < H_IN.to(tl.int32))
        # Base offset for this (n, ic, hi, :)
        base_ncih = (((n * C_IN + ic_i64) * H_IN) + hi.to(tl.int64)) * W_IN

        for s in range(K):
            wi = w_offsets * STRIDE - PADDING + s  # vector
            in_bounds_w = (wi >= 0) & (wi < W_IN.to(tl.int32))
            mask = w_mask & hi_in & in_bounds_w

            # Load input values
            wi_i64 = wi.to(tl.int64)
            x_offsets = base_ncih + wi_i64
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0).to(tl.float32)

            # Load weight scalar for (oc, r, s)
            w_offset = w_oc_base + (r * K + s)
            w_val = tl.load(w_ptr + w_offset).to(tl.float32)

            acc += x_vals * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + oc_i64).to(tl.float32)
        acc += b_val

    # Store output
    y_base = (((n * C_OUT + oc_i64) * H_OUT) + ho.to(tl.int64)) * W_OUT_i64
    y_offsets = y_base + w_offsets.to(tl.int64)
    tl.store(y_ptr + y_offsets, acc, mask=w_mask)


def _depthwise_conv2d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int, padding: int) -> torch.Tensor:
    # Fallbacks for unsupported conditions
    if x.device.type != 'cuda':
        return torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, groups=x.shape[1])

    # Expect weight of shape [C_out, 1, K, K] for groups=in_channels
    C_out, C_per_group, K, K2 = weight.shape
    assert K == K2, "Only square kernels are supported"
    C_in = x.shape[1]
    # If not depthwise as expected, fallback
    if C_per_group != 1 or (C_out % C_in) != 0:
        return torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, groups=C_in)

    N, _, H_in, W_in = x.shape
    ocpg = C_out // C_in

    # Output dims (no dilation)
    H_out = (H_in + 2 * padding - K) // stride + 1
    W_out = (W_in + 2 * padding - K) // stride + 1

    # Ensure contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else None

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Launch kernel
    grid = (triton.cdiv(W_out, 128), H_out, N * C_out)
    has_bias = bias is not None

    _dwconv2d_kernel[grid](
        x_c, w_c, (b_c if has_bias else x_c),  # dummy ptr if no bias, not used
        y,
        N,
        C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        stride, padding, ocpg,
        K=K,
        HAS_BIAS=has_bias,
        # Do not pass W_OUT again as keyword to avoid duplicate binding
    )
    return y


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep PyTorch Conv2d to hold parameters with identical initialization
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use Triton fast path on CUDA, fallback otherwise
        stride = self.conv2d.stride[0] if isinstance(self.conv2d.stride, tuple) else self.conv2d.stride
        padding = self.conv2d.padding[0] if isinstance(self.conv2d.padding, tuple) else self.conv2d.padding
        if x.is_cuda:
            return _depthwise_conv2d_triton(x, self.conv2d.weight, self.conv2d.bias, stride=stride, padding=padding)
        else:
            return self.conv2d(x)


# Test code
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]