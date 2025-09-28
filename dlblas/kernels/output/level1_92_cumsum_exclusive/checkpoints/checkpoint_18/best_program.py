# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * n_elements
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Perform parallel prefix sum
    inclusive = tl.associative_scan(x, 0, tl.add)
    exclusive = inclusive - x
    
    tl.store(output_ptr + offsets, exclusive, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.contiguous()
        output = torch.empty_like(x)
        n_elements = x.size(self.dim)
        batch_size = x.shape[0]
        
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        grid = (batch_size,)
        
        exclusive_cumsum_kernel[grid](x, output, n_elements, BLOCK_SIZE)
        return output

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================