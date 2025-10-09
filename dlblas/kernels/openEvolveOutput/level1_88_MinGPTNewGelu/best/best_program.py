# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Constants for GELU calculation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    k = 0.044715
    
    # Optimized GELU computation using sigmoid approximation
    x_cubed = x * x * x
    inner = x + k * x_cubed
    inner = inner * sqrt_2_over_pi
    
    # Compute tanh using optimized sigmoid: tanh(x) = 2*sigmoid(2x) - 1
    tanh_val = 2 * tl.sigmoid(2 * inner) - 1
    
    output = 0.5 * x * (1.0 + tanh_val)
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        n_elements = x.numel()
        if n_elements == 0:
            return x
        output = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================