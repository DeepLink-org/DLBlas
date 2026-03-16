import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _instancenorm_divide_2d_fused_kernel(
    x_ptr,  # input/output
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    eps, div_const,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # each program handles one (n, c)
    n = pid // C
    c = pid % C

    # Flattened spatial index [0, H*W)
    offs = tl.arange(0, BLOCK_HW)
    hw = H * W
    mask = offs < hw

    # Because the input is made contiguous by the caller, spatial slice is contiguous in memory.
    base = n * stride_n + c * stride_c
    ptrs = x_ptr + base + offs

    # Load values (masked), accumulate sum and sumsq in fp32
    x_vals = tl.load(ptrs, mask=mask, other=0.0)
    x_f32 = x_vals.to(tl.float32)

    sum_x = tl.sum(x_f32, axis=0)
    sum_x2 = tl.sum(x_f32 * x_f32, axis=0)

    # Compute mean and variance (population variance)
    inv_hw = tl.full((), 1.0 / hw, tl.float32)
    mean = sum_x * inv_hw
    var = sum_x2 * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)

    inv_std = tl.rsqrt(var + eps)
    rcp_div = 1.0 / div_const
    scale = inv_std * rcp_div
    bias = -mean * scale

    # Normalize and divide-by in one pass; store back
    y = tl.fma(x_f32, scale, bias)
    tl.store(ptrs, y.to(x_vals.dtype), mask=mask)


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    This version fuses InstanceNorm (no affine, no running stats) and the final division into a single
    Triton kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep the module to mirror original semantics; use its eps for correctness.
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        # If not CUDA or unsupported dtype, fall back to PyTorch reference ops
        if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.float32)):
            x = self.instance_norm(x)
            x = x / self.divide_by
            return x

        N, C, H, W = x.shape
        # Ensure contiguous layout for predictable strides
        x = x.contiguous()
        stride_n, stride_c, stride_h, stride_w = x.stride()

        # Triton kernel: one program per (n, c), vectorize across H*W
        HW = H * W
        BLOCK_HW = _next_power_of_two(HW)
        grid = (N * C,)

        eps = float(self.instance_norm.eps)
        div_const = float(self.divide_by)

        _instancenorm_divide_2d_fused_kernel[grid](
            x,  # in-place
            N, C, H, W,
            stride_n, stride_c, stride_h, stride_w,
            eps, div_const,
            BLOCK_HW=BLOCK_HW,
            num_warps=8,
            num_stages=4,
        )
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]