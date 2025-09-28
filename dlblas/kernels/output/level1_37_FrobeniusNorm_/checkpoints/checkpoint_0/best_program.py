# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _partial_sum_squares(
    x_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    squares = x * x
    block_sum = tl.sum(squares, axis=0)
    tl.store(partial_sums_ptr + pid, block_sum)

@triton.jit
def _normalize_kernel(
    x_ptr,
    norm,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x / norm
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        
        x_flat = x.contiguous().view(-1)
        total_elements = x_flat.numel()
        BLOCK_SIZE = 4096
        num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        partial_sums = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
        
        grid = (num_blocks,)
        _partial_sum_squares[grid](
            x_flat, 
            partial_sums, 
            total_elements, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        total_sum = torch.sum(partial_sums)
        norm_val = torch.sqrt(total_sum)
        
        output_flat = torch.empty_like(x_flat)
        _normalize_kernel[grid](
            x_flat, 
            norm_val, 
            output_flat, 
            total_elements, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output_flat.view_as(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================