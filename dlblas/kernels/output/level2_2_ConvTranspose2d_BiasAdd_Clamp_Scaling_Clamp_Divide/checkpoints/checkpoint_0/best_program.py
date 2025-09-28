# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _pointwise_ops(
    x_ptr,
    bias_ptr,
    output_ptr,
    scaling_factor,
    batch_size, 
    channels, 
    height, 
    width,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    if pid_b >= batch_size or pid_c >= channels:
        return

    # Calculate base offset for current batch and channel
    base_offset = pid_b * channels * height * width + pid_c * height * width
    
    # Spatial indices processing
    idx_start = pid_s * BLOCK_SIZE
    offsets = idx_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (height * width)
    
    # Load bias once per channel
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Load input data
    x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
    
    # Fused pointwise operations
    x_vals = x_vals + bias_val
    x_vals = tl.minimum(tl.maximum(x_vals, 0.0), 1.0)
    x_vals = x_vals * scaling_factor
    x_vals = tl.minimum(tl.maximum(x_vals, 0.0), 1.0)
    x_vals = x_vals / scaling_factor
    
    # Store results
    tl.store(output_ptr + base_offset + offsets, x_vals, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        batch_size, channels, height, width = x.shape
        
        # Ensure contiguous memory layout
        x_contig = x.contiguous()
        output = torch.empty_like(x_contig)
        bias_1d = self.bias.view(-1)
        
        # Configure kernel launch parameters
        BLOCK = 256
        num_spatial = height * width
        num_blocks = (num_spatial + BLOCK - 1) // BLOCK
        grid = (batch_size, channels, num_blocks)
        
        # Launch optimized Triton kernel
        _pointwise_ops[grid](
            x_contig, bias_1d, output, self.scaling_factor,
            batch_size, channels, height, width,
            BLOCK_SIZE=BLOCK
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================