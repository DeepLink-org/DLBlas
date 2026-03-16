import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _mish_mish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Softplus with PyTorch's default threshold=20 for numerical parity:
    # softplus(x) = x if x > 20, ~exp(x) if x < -20, else max(x,0)+log1p(exp(-|x|))
    abs_x = tl.abs(x32)
    sp1_mid = tl.maximum(x32, 0.0) + libdevice.log1p(tl.exp(-abs_x))
    sp1 = tl.where(x32 > 20.0, x32, tl.where(x32 < -20.0, tl.exp(x32), sp1_mid))

    # Use fast libdevice tanh for better performance
    tanh_sp1 = libdevice.tanh(sp1)
    mish1 = x32 * tanh_sp1

    # Second Mish
    abs_m1 = tl.abs(mish1)
    sp2_mid = tl.maximum(mish1, 0.0) + libdevice.log1p(tl.exp(-abs_m1))
    sp2 = tl.where(mish1 > 20.0, mish1, tl.where(mish1 < -20.0, tl.exp(mish1), sp2_mid))

    tanh_sp2 = libdevice.tanh(sp2)
    out32 = mish1 * tanh_sp2

    out = out32.to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)


def mish_mish_triton(x: torch.Tensor) -> torch.Tensor:
    # Fallback to torch if not CUDA or Triton-incompatible dtype or requires grad
    if (not x.is_cuda) or x.requires_grad or x.numel() == 0:
        return torch.nn.functional.mish(torch.nn.functional.mish(x))

    # Support common dtypes; compute in fp32 internally for stability
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.nn.functional.mish(torch.nn.functional.mish(x))

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _mish_mish_kernel[grid](x_contig, y, n_elements)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    The two Mish activations are fused into a single Triton kernel for speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Use fused Triton kernel when possible; otherwise fall back to PyTorch ops
        x = mish_mish_triton(x)
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