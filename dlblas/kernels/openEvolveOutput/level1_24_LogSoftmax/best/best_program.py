# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _log_softmax_kernel(
    x_ptr,
    output_ptr,
    stride_x,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * stride_x
    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    
    # First pass: Compute row max
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        row = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(row, 0))
    
    # Second pass: Compute row sum of exponentials
    row_sum = tl.zeros((), dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        row = tl.load(x_ptr + offsets, mask=mask)
        shifted = row - row_max
        exp_row = tl.exp(shifted)
        row_sum += tl.sum(exp_row, 0)
    
    log_sum = tl.log(row_sum)
    
    # Third pass: Compute log softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        row = tl.load(x_ptr + offsets, mask=mask)
        output = row - row_max - log_sum
        tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            return torch.log_softmax(x, dim=self.dim)
            
        x = x.contiguous()
        output = torch.empty_like(x)
        n_rows, n_cols = x.shape
        grid = (n_rows,)
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
        _log_softmax_kernel[grid](x, output, x.stride(0), n_cols, BLOCK_SIZE)
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================