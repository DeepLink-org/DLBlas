# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    base = row_idx * n
    running_sum = 0.0
    for i in range(n-1, -1, -1):
        val = tl.load(x_ptr + base + i)
        running_sum += val
        tl.store(out_ptr + base + i, running_sum)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_contig = x.contiguous()
        shape = x_contig.shape
        dim_size = shape[self.dim]
        
        # Flatten all dimensions except the target dimension
        x_flat = x_contig.view(-1, dim_size)
        out_flat = torch.empty_like(x_flat)
        
        n_rows = x_flat.size(0)
        grid = (n_rows,)
        reverse_cumsum_kernel[grid](x_flat, out_flat, dim_size, BLOCK_SIZE=1)
        
        return out_flat.view(shape)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================