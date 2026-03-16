import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_sub_mish_kernel(
    x_ptr,          # in-place pointer to tensor
    n_elements,     # total number of elements
    sub1,           # subtract_value_1 (scalar)
    sub2,           # subtract_value_2 (scalar)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and upcast for numerics
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Apply sequential subtractions: (x - sub1) - sub2
    x32 = x32 - sub1
    x32 = x32 - sub2

    # Stable softplus: max(x, 0) + log1p(exp(-|x|))
    abs_x = tl.abs(x32)
    m = tl.maximum(x32, 0.0)
    sp = m + libdevice.log1p(tl.exp(-abs_x))

    # tanh(sp) via sigmoid for stability: tanh(s) = 2 * sigmoid(2s) - 1
    neg2sp = -2.0 * sp
    s2 = 1.0 / (1.0 + tl.exp(neg2sp))
    t = 2.0 * s2 - 1.0

    y32 = x32 * t
    y = y32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def _fused_sub_mish_inplace(x: torch.Tensor, sub1: float, sub2: float) -> torch.Tensor:
    # Fused: (x - sub1 - sub2) -> mish, computed in-place on CUDA tensor
    n_elements = x.numel()
    if n_elements == 0:
        return x
    x = x.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_sub_mish_kernel[grid](x, n_elements, float(sub1), float(sub2),
                                 BLOCK_SIZE=4096, num_warps=8, num_stages=1)
    return x


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        x = self.conv(x)
        if x.is_cuda:
            x = _fused_sub_mish_inplace(x, self.subtract_value_1, self.subtract_value_2)
        else:
            x = x - self.subtract_value_1
            x = x - self.subtract_value_2
            x = torch.nn.functional.mish(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]