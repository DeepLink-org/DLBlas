# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _pointwise_fusion(
    input_ptr,
    scaling_factor_ptr,
    bias_ptr,
    output_ptr,
    total_elements,
    out_channels,
    channel_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Calculate channel indices
    channel_idx = (offs // channel_stride) % out_channels
    
    # Load input and parameters
    x = tl.load(input_ptr + offs, mask=mask)
    s = tl.load(scaling_factor_ptr + channel_idx, mask=mask)
    b = tl.load(bias_ptr + channel_idx, mask=mask)
    
    # Fused operations with mathematical equivalents
    x = x * s
    
    # Replace tl.tanh(x) with (exp(2*x)-1)/(exp(2*x)+1)
    exp_2x = tl.exp(2 * x)
    x = (exp_2x - 1) / (exp_2x + 1)
    
    x = x * b
    
    # Replace tl.sigmoid(x) with 1/(1+exp(-x))
    x = 1 / (1 + tl.exp(-x))
    
    tl.store(output_ptr + offs, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        x = self.conv(x)
        
        # Prepare parameters and tensors
        scaling_factor_flat = self.scaling_factor.view(-1)
        bias_flat = self.bias.view(-1)
        channel_stride = x.shape[2] * x.shape[3] * x.shape[4]
        total_elements = x.numel()
        output = torch.empty_like(x)
        
        # Launch Triton kernel
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        _pointwise_fusion[grid](
            x.view(-1),
            scaling_factor_flat,
            bias_flat,
            output.view(-1),
            total_elements,
            x.shape[1],
            channel_stride,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
# =================== EVOLVE-BLOCK-END ===================