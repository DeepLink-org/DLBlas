# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _forward_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    depth, height, width,
    in_channels,
    input_bs_stride, input_c_stride, input_d_stride, input_h_stride, input_w_stride,
    output_bs_stride, output_c_stride, output_d_stride, output_h_stride, output_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # 3D spatial position processing
    pid_bd = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Reconstruct batch and depth indices
    batch_idx = pid_bd // depth
    d_idx = pid_bd % depth
    
    # Check boundaries
    if batch_idx >= batch_size or d_idx >= depth or pid_h >= height or pid_w >= width:
        return

    # Initialize reduction values
    max_val = -float('inf')
    sum_exp = 0.0
    hard_swish_val = 0.0
    max_output = -float('inf')
    
    # Process channels in blocks
    for c in range(0, in_channels, BLOCK_SIZE):
        c_offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < in_channels
        
        # Calculate input pointer offsets
        input_offsets = (
            batch_idx * input_bs_stride + 
            c_offsets * input_c_stride + 
            d_idx * input_d_stride + 
            pid_h * input_h_stride + 
            pid_w * input_w_stride
        )
        inputs = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # LogSumExp reduction
        local_max = tl.max(inputs, axis=0)
        if local_max > max_val:
            max_val = local_max
        
        # Compute exp(values - max_val) for numerical stability
        exp_vals = tl.exp(inputs - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Final LogSumExp calculation
    lse = max_val + tl.log(sum_exp)
    
    # HardSwish: x * sigmoid(x+3)/6
    sigmoid_val = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hard_swish_val = lse * sigmoid_val / 6.0
    
    # Process bias and max reduction
    for c in range(0, in_channels, BLOCK_SIZE):
        c_offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < in_channels
        
        # Load bias values
        bias_vals = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0)
        
        # Subtract bias and clamp
        candidate = hard_swish_val - bias_vals
        candidate = tl.minimum(tl.maximum(candidate, -1.0), 1.0)
        
        # Max reduction
        local_max = tl.max(candidate, axis=0)
        if local_max > max_output:
            max_output = local_max
    
    # Calculate output pointer offset
    output_offset = (
        batch_idx * output_bs_stride + 
        0 * output_c_stride + 
        d_idx * output_d_stride + 
        pid_h * output_h_stride + 
        pid_w * output_w_stride
    )
    tl.store(output_ptr + output_offset, max_output)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        batch, channels, depth, height, width = x.shape
        
        # Prepare output tensor
        output = torch.empty(batch, 1, depth, height, width, device=x.device, dtype=x.dtype)
        
        # Compute 3D grid dimensions
        grid_bd = batch * depth
        grid = (grid_bd, height, width)
        
        # Get tensor strides
        strides = x.stride()
        out_strides = output.stride()
        
        # Launch Triton kernel
        BLOCK_SIZE = 16
        _forward_kernel[grid](
            x,
            self.bias.view(-1),
            output,
            batch, depth, height, width, channels,
            strides[0], strides[1], strides[2], strides[3], strides[4],
            out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
            BLOCK_SIZE
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================