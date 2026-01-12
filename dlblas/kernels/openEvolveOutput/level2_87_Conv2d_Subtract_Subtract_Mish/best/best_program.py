# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _pointwise_mish(
    x_ptr,
    output_ptr,
    s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    x = x - s
    softplus = tl.log(1 + tl.exp(x))
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    tanh_softplus = (tl.exp(2*softplus) - 1) / (tl.exp(2*softplus) + 1)
    mish_val = x * tanh_softplus
    tl.store(output_ptr + offset, mish_val, mask=mask)

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
        n_elements = x.numel()
        s = self.subtract_value_1 + self.subtract_value_2
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _pointwise_mish[grid](x, output, s, n_elements)
        return output

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
# =================== EVOLVE-BLOCK-END ===================