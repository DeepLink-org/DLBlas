# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_tanh_scale_bias_kernel(
    input_ptr,
    output_ptr,
    bias_ptr,
    scaling_factor,
    batch_size,
    out_channels,
    H,
    W,
    stride_batch,
    stride_channel,
    stride_height,
    stride_width,
    BLOCK_SIZE: tl.constexpr
):
    # Compute program IDs
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Calculate batch and channel indices
    batch_idx = pid_bc // out_channels
    channel_idx = pid_bc % out_channels
    
    # Compute base pointer offsets
    base_offset = batch_idx * stride_batch + channel_idx * stride_channel + pid_h * stride_height
    offsets = base_offset + tl.arange(0, BLOCK_SIZE) * stride_width
    
    # Create mask for valid columns
    mask = tl.arange(0, BLOCK_SIZE) < W
    
    # Load input and bias
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Compute tanh using stable exponential method
    abs_x = tl.abs(x)
    sign = tl.where(x >= 0, 1.0, -1.0)
    t = tl.exp(-2.0 * abs_x)
    tanh_x = sign * (1.0 - t) / (1.0 + t)
    
    # Fused operations: tanh(x)*scaling_factor + bias
    y = tanh_x * scaling_factor + bias_val
    
    # Store results
    tl.store(output_ptr + offsets, y, mask=mask)

def fused_tanh_scale_bias(x, scaling_factor, bias):
    # Ensure contiguous tensors
    x_contig = x.contiguous()
    bias_contig = bias.view(-1).contiguous()
    output = torch.empty_like(x_contig)
    
    # Get tensor dimensions
    batch_size, out_channels, H, W = x_contig.shape
    strides = (x_contig.stride(0), x_contig.stride(1), 
               x_contig.stride(2), x_contig.stride(3))
    
    # Compute block size and grid
    BLOCK_SIZE = triton.next_power_of_2(W)
    grid = (batch_size * out_channels, H)
    
    # Launch kernel
    fused_tanh_scale_bias_kernel[grid](
        x_contig, output, bias_contig,
        scaling_factor,
        batch_size,
        out_channels,
        H,
        W,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Fused tanh, scaling and bias using Triton
        x = fused_tanh_scale_bias(x, self.scaling_factor, self.bias)
        # Max-pooling
        x = self.max_pool(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================