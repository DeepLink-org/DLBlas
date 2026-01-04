# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_scan_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
):
    row_idx = tl.program_id(0)
    input_start = row_idx * input_row_stride
    output_start = row_idx * output_row_stride
    
    # Store initial 0 for exclusive cumsum
    tl.store(output_ptr + output_start, 0.0)
    cum = 0.0
    for j in range(0, n_cols):
        x_val = tl.load(x_ptr + input_start + j)
        cum += x_val
        tl.store(output_ptr + output_start + j + 1, cum)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape[0], x.shape[1]
        output = torch.empty((n_rows - 1, n_cols + 1), device=x.device, dtype=x.dtype)
        
        grid = (n_rows - 1,)
        exclusive_scan_kernel[grid](
            x, output, n_cols, x.stride(0), output.stride(0)
        )
        return output

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================