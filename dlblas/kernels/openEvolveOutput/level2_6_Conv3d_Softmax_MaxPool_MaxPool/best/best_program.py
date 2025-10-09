# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    n_rows,
    BLOCK_COL: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_COL)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    max_val = tl.max(row, axis=0)
    row = row - max_val
    exp_row = tl.exp(row)
    sum_val = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_val
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_row, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        
        # Reshape for channel-wise softmax
        N, C, D, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_2d = x_perm.view(-1, C)
        output_2d = torch.empty_like(x_2d)
        
        # Launch Triton kernel
        n_rows = x_2d.shape[0]
        grid = (n_rows,)
        softmax_kernel[grid](
            output_2d, x_2d, C, 
            x_2d.stride(0), output_2d.stride(0),
            n_rows, BLOCK_COL=16
        )
        
        # Restore original shape
        output_perm = output_2d.view(N, D, H, W, C)
        x = output_perm.permute(0, 4, 1, 2, 3).contiguous()
        
        x = self.pool1(x)
        x = self.pool2(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================