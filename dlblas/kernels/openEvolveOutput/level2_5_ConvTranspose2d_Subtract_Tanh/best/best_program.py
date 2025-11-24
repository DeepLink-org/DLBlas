# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_bias_tanh_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    out_channels,
    height_width,
    stride_batch,
    stride_channel,
    stride_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    batch_idx = pid_bc // out_channels
    channel_idx = pid_bc % out_channels
    
    spatial_start = pid_spatial * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < height_width
    
    # Compute base pointers
    batch_offset = batch_idx * stride_batch
    channel_offset = channel_idx * stride_channel
    base_ptr = input_ptr + batch_offset + channel_offset
    
    # Load bias for current channel
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Load input block
    input_ptrs = base_ptr + spatial_offsets
    x = tl.load(input_ptrs, mask=spatial_mask, other=0.0)
    
    # Compute fused operations: subtract bias then tanh using fast approximation
    y = x - bias_val
    y = 2 * tl.sigmoid(2 * y) - 1  # tanh approximation: 2σ(2x)-1
    
    # Store result
    output_ptrs = output_ptr + batch_offset + channel_offset + spatial_offsets
    tl.store(output_ptrs, y, mask=spatial_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        batch, channels, height, width = x.shape
        
        # Prepare for Triton kernel
        x_cont = x.contiguous()
        output = torch.empty_like(x_cont)
        height_width = height * width
        
        # Compute strides for contiguous tensor
        stride_batch = channels * height_width
        stride_channel = height_width
        stride_spatial = 1
        
        # Configure kernel grid
        grid_bc = batch * self.out_channels
        grid_spatial = triton.cdiv(height_width, 1024)
        grid = (grid_bc, grid_spatial)
        
        # Launch kernel
        fused_bias_tanh_kernel[grid](
            x_cont,
            self.bias.view(-1),
            output,
            self.out_channels,
            height_width,
            stride_batch,
            stride_channel,
            stride_spatial,
            BLOCK_SIZE=1024
        )
        
        return output

batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# =================== EVOLVE-BLOCK-END ===================