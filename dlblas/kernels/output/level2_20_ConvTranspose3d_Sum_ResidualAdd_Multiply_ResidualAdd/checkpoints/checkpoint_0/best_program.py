# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_operations_kernel(
    x0_ptr,
    bias_ptr,
    output_ptr,
    num_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    # Extract program IDs
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_block = tl.program_id(2)
    
    # Compute spatial offsets
    spatial_offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < num_spatial
    
    # Calculate base pointer offset for current batch and channel
    base_offset = pid_batch * num_spatial + spatial_offsets
    x0_vals = tl.load(x0_ptr + base_offset, mask=mask, other=0.0)
    
    # Load bias value for current channel
    bias_val = tl.load(bias_ptr + pid_channel)
    
    # Fused computation: 2*x^2 + (bias+1)*x
    result = 2 * (x0_vals * x0_vals) + (bias_val + 1.0) * x0_vals
    tl.store(output_ptr + base_offset, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                                stride=stride, padding=padding, 
                                                output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
    def forward(self, x):
        x0 = self.conv_transpose(x)
        batch_size, out_channels, depth, height, width = x0.shape
        num_spatial = depth * height * width
        output = torch.empty_like(x0)
        
        # Reshape tensors for kernel processing
        x0_flat = x0.view(batch_size, out_channels, -1)
        output_flat = output.view(batch_size, out_channels, -1)
        bias_flat = self.bias.view(out_channels)
        
        # Configure kernel grid and launch
        BLOCK_SIZE = 1024
        grid = (batch_size, out_channels, triton.cdiv(num_spatial, BLOCK_SIZE))
        fused_operations_kernel[grid](
            x0_flat, bias_flat, output_flat, 
            num_spatial, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output_flat.view_as(x0)

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================