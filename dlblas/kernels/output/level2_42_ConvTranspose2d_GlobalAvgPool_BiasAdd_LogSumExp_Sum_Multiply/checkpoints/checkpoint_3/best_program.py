# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_operation_kernel(
    # Input pointers
    conv_output_ptr, 
    bias_ptr,
    # Output pointer
    output_ptr,
    # Tensor dimensions
    batch_size, 
    channels, 
    height, 
    width,
    # Stride information
    stride_conv_b, stride_conv_c, stride_conv_h, stride_conv_w,
    # Blocking
    BLOCK_SIZE: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    # Batch index
    pid_batch = tl.program_id(0)
    
    # Channel and spatial reduction
    spatial_size = height * width
    channel_vals = tl.zeros((channels,), dtype=tl.float32)
    
    # Reduce spatial dimensions per channel
    for c in range(0, channels):
        spatial_sum = 0.0
        # Reduce spatial dimensions in blocks
        for i in range(0, spatial_size, REDUCE_BLOCK):
            offsets = i + tl.arange(0, REDUCE_BLOCK)
            mask = offsets < spatial_size
            
            # Calculate spatial indices
            h_idx = offsets // width
            w_idx = offsets % width
            
            # Calculate pointers
            ptr = (
                conv_output_ptr + 
                pid_batch * stride_conv_b + 
                c * stride_conv_c + 
                h_idx * stride_conv_h + 
                w_idx * stride_conv_w
            )
            
            # Load data
            data = tl.load(ptr, mask=mask, other=0.0)
            spatial_sum += tl.sum(data, axis=0)
        
        # Store channel average
        channel_vals = tl.store(channel_vals, [c], spatial_sum / spatial_size)
    
    # Load bias and add
    bias = tl.load(bias_ptr + tl.arange(0, channels))
    channel_vals += bias
    
    # Log-sum-exp reduction
    max_val = tl.max(channel_vals, axis=0)
    exp_vals = tl.exp(channel_vals - max_val)
    exp_sum = tl.sum(exp_vals, axis=0)
    result = max_val + tl.log(exp_sum)
    
    # Final multiplication and store
    result *= 10.0
    tl.store(output_ptr + pid_batch, result)

class ModelNew(nn.Module):
    """
    Optimized model using Triton for fused operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        # Run transposed convolution
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        
        # Prepare output tensor
        output = torch.empty(B, device=x.device, dtype=torch.float32)
        
        # Grid and kernel configuration
        grid = (B,)
        fused_operation_kernel[grid](
            x,
            self.bias,
            output,
            B,
            C,
            H,
            W,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_SIZE=128,
            REDUCE_BLOCK=128,
        )
        
        return output.unsqueeze(1)  # Maintain [B, 1] output shape

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# =================== EVOLVE-BLOCK-END ===================