import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gelu_gap2d_fused_row_kernel(
    x_ptr,                      # *f32/ *f16 input tensor pointer [N, C, H, W]
    y_ptr,                      # *f32 output tensor pointer [N, C]
    C, H, W,                    # ints
    stride_n, stride_c, stride_h, stride_w,  # strides for x in elements
    out_stride_n, out_stride_c,              # strides for y in elements
    BLOCK_W: tl.constexpr,                   # tile size across flattened H*W
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    # base pointer for this (n, c) plane
    base = n * stride_n + c * stride_c

    # Flatten spatial dims; host makes x contiguous so H*W is contiguous
    total_hw = H * W
    idx = tl.arange(0, BLOCK_W)

    # Vector accumulator to minimize per-iteration reductions
    acc_vec = tl.zeros((BLOCK_W,), dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)

    # Tile over flattened plane
    for start in range(0, total_hw, BLOCK_W):
        offs = base + start + idx
        mask = (start + idx) < total_hw
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_vals = 0.5 * vals * (1.0 + libdevice.erf(vals * inv_sqrt2))
        gelu_vals = tl.where(mask, gelu_vals, 0.0)
        acc_vec += gelu_vals

    acc = tl.sum(acc_vec, axis=0)
    mean_val = acc / tl.full((), total_hw, dtype=tl.float32)

    out_off = n * out_stride_n + c * out_stride_c
    tl.store(y_ptr + out_off, mean_val)


def gelu_global_avg_pool2d_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Fused GELU + global average pooling over H and W using Triton.
    Input:  x of shape (N, C, H, W)
    Output: y of shape (N, C)
    """
    assert x.dim() == 4
    if not x.is_cuda:
        y = torch.nn.functional.gelu(x)
        y = torch.nn.functional.adaptive_avg_pool2d(y, 1)
        return y.squeeze(-1).squeeze(-1)

    # Use float32 accumulation for numerical correctness
    orig_dtype = x.dtype
    x_fp32 = x if orig_dtype == torch.float32 else x.float()
    x_fp32 = x_fp32.contiguous()

    N, C, H, W = x_fp32.shape
    y = torch.empty((N, C), device=x_fp32.device, dtype=torch.float32)

    # Choose BLOCK size to minimize loop iterations and reduction overhead
    total_hw = H * W
    if total_hw >= 4096:
        BLOCK_W = 1024
        num_warps = 8
        num_stages = 3
    elif total_hw >= 1024:
        BLOCK_W = 1024
        num_warps = 4
        num_stages = 2
    elif total_hw >= 512:
        BLOCK_W = 512
        num_warps = 4
        num_stages = 2
    elif total_hw >= 256:
        BLOCK_W = 256
        num_warps = 2
        num_stages = 2
    else:
        BLOCK_W = 128
        num_warps = 2
        num_stages = 1

    grid = (N * C,)
    _gelu_gap2d_fused_row_kernel[grid](
        x_fp32,
        y,
        C, H, W,
        x_fp32.stride(0), x_fp32.stride(1), x_fp32.stride(2), x_fp32.stride(3),
        y.stride(0), y.stride(1),
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if orig_dtype != torch.float32:
        y = y.to(orig_dtype)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        # Use Triton fused GELU + Global AvgPool on CUDA for speed; fallback elsewhere
        if x.is_cuda:
            x = gelu_global_avg_pool2d_triton(x)
        else:
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]