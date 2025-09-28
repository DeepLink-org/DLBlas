# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_softmax_mean_kernel(
    input_ptr,
    output_ptr,
    n_spatial,
    n_channels,
    spatial_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Get current program IDs
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Create pointers for the current channel in the input
    channel_offset = pid_batch * spatial_stride + pid_channel
    input_ptrs = input_ptr + channel_offset + tl.arange(0, BLOCK_SIZE) * n_channels
    
    # Mask for valid spatial locations
    spatial_idx = tl.arange(0, BLOCK_SIZE)
    mask = spatial_idx < n_spatial
    
    # Load data for current channel across spatial locations
    data = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Apply HardSwish activation: x * relu6(x+3) / 6
    shifted = data + 3.0
    clipped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    activated = data * clipped / 6.0
    
    # Compute softmax denominator per spatial location
    max_val = tl.max(activated, axis=0)
    exp_vals = tl.exp(activated - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax for each spatial location
    softmax_vals = exp_vals / sum_exp
    
    # Compute mean across spatial locations
    mean_val = tl.sum(softmax_vals, axis=0) / n_spatial
    
    # Store the result
    output_offset = pid_batch * n_channels + pid_channel
    tl.store(output_ptr + output_offset, mean_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton for fused activation, softmax and mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.n_spatial = None  # Will be set in first forward pass

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Get tensor dimensions
        B, C, D, H, W = x.shape
        n_spatial = D * H * W
        
        # Flatten spatial dimensions and permute to [B, spatial, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, n_spatial, C)
        
        # Initialize output tensor
        output = torch.empty((B, C), device=x.device, dtype=x.dtype)
        
        # Set kernel parameters
        spatial_stride = n_spatial * C
        grid = (B, C)
        
        # Determine block size (next power of two for spatial dim)
        BLOCK_SIZE = triton.next_power_of_2(n_spatial)
        
        # Launch kernel
        fused_softmax_mean_kernel[grid](
            x, output, n_spatial, C, spatial_stride, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================