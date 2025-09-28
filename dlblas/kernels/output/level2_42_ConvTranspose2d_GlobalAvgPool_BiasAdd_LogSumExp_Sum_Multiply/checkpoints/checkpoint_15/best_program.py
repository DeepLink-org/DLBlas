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
    height,
    width,
    # Stride information
    stride_b, stride_c, stride_h, stride_w,
    # Blocking and channels
    BLOCK_SIZE: tl.constexpr,
    channels: tl.constexpr,
):
    # Batch index
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    spatial_size = height * width
    max_val = -float('inf')
    exp_sum = 0.0
    
    # Process each channel
    for c in range(channels):
        spatial_sum = 0.0
        # Process spatial dimension in blocks
        for i in range(0, spatial_size, BLOCK_SIZE):
            # Create spatial offsets and mask
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < spatial_size
            
            # Convert to 2D indices
            h_idx = offsets // width
            w_idx = offsets % width
            
            # Compute pointer and load data
            ptr = (
                conv_output_ptr +
                pid * stride_b +
                c * stride_c +
                h_idx * stride_h +
                w_idx * stride_w
            )
            data = tl.load(ptr, mask=mask, other=0.0)
            spatial_sum += tl.sum(data, axis=0)
        
        # Compute channel average and add bias
        channel_avg = spatial_sum / spatial_size
        bias_val = tl.load(bias_ptr + c)
        val = channel_avg + bias_val
        
        # Update log-sum-exp state
        if max_val == -float('inf'):
            max_val = val
            exp_sum = 1.0
        elif val > max_val:
            exp_sum = exp_sum * tl.exp(max_val - val) + 1.0
            max_val = val
        else:
            exp_sum += tl.exp(val - max_val)
    
    # Compute final result
    result = (tl.log(exp_sum) + max_val) * 10.0
    tl.store(output_ptr + pid, result)

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
        
        # Ensure contiguous memory
        x = x.contiguous()
        
        # Grid and kernel configuration
        grid = (B,)
        fused_operation_kernel[grid](
            x,
            self.bias.view(-1),
            output,
            B,
            H, W,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_SIZE=128,
            channels=C,
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