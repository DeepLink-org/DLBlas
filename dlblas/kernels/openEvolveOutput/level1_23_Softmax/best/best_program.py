# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    max_val = tl.full((1,), float('-inf'), tl.float32)
    sum_exp = tl.zeros((1,), tl.float32)
    
    # Find max value in row
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask, -float('inf'))
        chunk_max = tl.max(x_vals, 0)
        max_val = tl.maximum(max_val, chunk_max)
    
    # Compute sum of exponentials
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask, 0)
        x_shifted = x_vals - max_val
        exp_vals = tl.exp(x_shifted)
        chunk_sum = tl.sum(exp_vals, 0)
        sum_exp += chunk_sum
    
    # Compute and store softmax
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask, 0)
        x_shifted = x_vals - max_val
        exp_vals = tl.exp(x_shifted)
        softmax_vals = exp_vals / sum_exp
        tl.store(output_ptr + row_start + col_offsets, softmax_vals, mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_rows, n_cols = x.shape
        BLOCK_SIZE = 2048  # Optimized for H100's 2048-bit memory interface
        grid = (n_rows,)
        softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE)
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================