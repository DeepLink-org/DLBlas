# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    min_val = tl.full((), float('inf'), dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = tl.arange(0, BLOCK_SIZE)
        mask = col_indices < (n_cols - col_offset)
        cols = tl.load(input_ptr + row_start + (col_offset + col_indices) * stride_col, 
                      mask=mask, other=float('inf'))
        current_min = tl.min(cols, axis=0)
        min_val = tl.minimum(min_val, current_min)
    tl.store(output_ptr + row_idx, min_val)

def triton_min(x, dim, keepdim=False):
    if x.dim() != 2 or dim != 1 or not keepdim:
        return torch.min(x, dim=dim, keepdim=keepdim)
    output = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)
    grid = (x.size(0),)
    n_cols = x.size(1)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    min_reduction_kernel[grid](
        x, output, n_cols, x.stride(0), x.stride(1), BLOCK_SIZE=BLOCK_SIZE
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = triton_min(x, dim=1, keepdim=True)
        x = x + self.bias
        return x

batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
# =================== EVOLVE-BLOCK-END ===================