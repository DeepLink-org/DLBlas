# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _l1norm_kernel(
    x_ptr,
    output_ptr,
    stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * stride_row
    
    row_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        abs_x = tl.abs(x_vals)
        chunk_sum = tl.sum(abs_x, axis=0)
        row_sum += chunk_sum

    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        normalized = x_vals / row_sum
        tl.store(output_ptr + row_start + cols, normalized, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        output = torch.empty_like(x)
        n_rows, n_cols = x.shape[0], x.shape[1]
        grid = (n_rows,)
        BLOCK_SIZE = 1024
        
        _l1norm_kernel[grid](
            x, output,
            stride_row=x.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================