# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _l2_norm_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # First pass: compute the squared norm
    norm_sq = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + offset
        mask = col_offsets < n_cols
        ptr = x_ptr + row_start + col_offsets
        vals = tl.load(ptr, mask=mask, other=0.0)
        norm_sq += tl.sum(vals * vals)
    
    norm = tl.sqrt(norm_sq)
    
    # Second pass: normalize and store
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + offset
        mask = col_offsets < n_cols
        ptr = x_ptr + row_start + col_offsets
        vals = tl.load(ptr, mask=mask, other=0.0)
        normalized = vals / norm
        tl.store(output_ptr + row_start + col_offsets, normalized, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        output = torch.empty_like(x)
        n_rows, n_cols = x.shape
        
        if n_cols == 0:
            return output
        
        grid = (n_rows,)
        BLOCK_SIZE = 1024
        _l2_norm_kernel[grid](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================