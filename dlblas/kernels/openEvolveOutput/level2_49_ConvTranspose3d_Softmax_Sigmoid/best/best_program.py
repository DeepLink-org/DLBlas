# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_softmax_sigmoid_kernel(
    input_ptr, 
    output_ptr,
    input_row_stride, 
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    output_row_start = row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(row, axis=0)
    row = row - max_val
    row = tl.exp(row)
    row_sum = tl.sum(row, axis=0)
    row = row / row_sum
    row = tl.sigmoid(row)
    tl.store(output_ptr + output_row_start + col_offsets, row, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        x_2d = x.reshape(-1, C)
        output_2d = torch.empty_like(x_2d)
        
        n_rows = x_2d.size(0)
        n_cols = x_2d.size(1)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        grid = (n_rows,)
        fused_softmax_sigmoid_kernel[grid](
            x_2d, output_2d,
            x_2d.stride(0), output_2d.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output_2d.view(B, C, D, H, W)

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
# =================== EVOLVE-BLOCK-END ===================