# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(x_ptr, output_ptr, n_cols, 
                          BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    carry = 0.0
    num_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    
    for block_idx in range(num_blocks-1, -1, -1):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        col_offsets = row_start + col_offsets
        
        # Load current block
        block = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
        
        # Compute reverse cumsum within block
        local_cumsum = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        for j in range(BLOCK_SIZE-1, -1, -1):
            if mask[j]:
                carry += block[j]
                local_cumsum = tl.where(
                    tl.arange(0, BLOCK_SIZE) == j, 
                    carry, 
                    local_cumsum
                )
        
        # Store computed block
        tl.store(output_ptr + col_offsets, local_cumsum, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if x.is_cuda:
            output = torch.empty_like(x)
            n_rows, n_cols = x.shape
            grid = (n_rows,)
            reverse_cumsum_kernel[grid](
                x, output, n_cols, 
                BLOCK_SIZE=1024
            )
            return output
        else:
            return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================