# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    temp_softmax_ptr,
    n_spatial,
    n_channels,
    BLOCK_SIZE_C: tl.constexpr
):
    # Get program IDs for batch and spatial location
    pid_batch = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    # Calculate base pointer for current spatial location
    base_ptr = pid_batch * n_spatial * n_channels + pid_spatial * n_channels
    offsets = base_ptr + tl.arange(0, BLOCK_SIZE_C)
    mask = tl.arange(0, BLOCK_SIZE_C) < n_channels
    
    # Load input data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardSwish activation
    shifted = data + 3.0
    clipped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    activated = data * clipped / 6.0
    activated = tl.maximum(activated, 0.0)  # ReLU equivalent
    
    # Compute softmax over channels
    max_val = tl.max(activated, axis=0)
    exp_vals = tl.exp(activated - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store results
    tl.store(temp_softmax_ptr + offsets, softmax_vals, mask=mask)

@triton.jit
def mean_kernel(
    temp_softmax_ptr,
    output_ptr,
    n_spatial,
    n_channels,
    BLOCK_SIZE: tl.constexpr
):
    # Get program IDs for batch and channel
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    output_offset = pid_batch * n_channels + pid_channel
    base_ptr = pid_batch * n_spatial * n_channels + pid_channel
    
    # Accumulate sum across spatial locations
    total = 0.0
    for idx in range(0, n_spatial, BLOCK_SIZE):
        offsets = base_ptr + (idx + tl.arange(0, BLOCK_SIZE)) * n_channels
        mask = (idx + tl.arange(0, BLOCK_SIZE)) < n_spatial
        vals = tl.load(temp_softmax_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)
    
    # Compute and store mean
    mean_val = total / n_spatial
    tl.store(output_ptr + output_offset, mean_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton for fused operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Get tensor dimensions
        B, C, D, H, W = x.shape
        n_spatial = D * H * W
        
        # Reshape tensor: [B, C, D, H, W] -> [B, n_spatial, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, n_spatial, C)
        
        # Allocate temporary tensor for softmax results
        temp_softmax = torch.empty((B, n_spatial, C), device=x.device, dtype=x.dtype)
        
        # Launch softmax kernel
        grid_softmax = (B, n_spatial)
        BLOCK_SIZE_C = triton.next_power_of_2(C)
        softmax_kernel[grid_softmax](
            x, temp_softmax, n_spatial, C, BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        
        # Allocate output tensor
        output = torch.empty((B, C), device=x.device, dtype=x.dtype)
        
        # Launch mean kernel
        grid_mean = (B, C)
        BLOCK_SIZE_SPATIAL = 1024  # Optimal block size for spatial reduction
        mean_kernel[grid_mean](
            temp_softmax, output, n_spatial, C, BLOCK_SIZE=BLOCK_SIZE_SPATIAL
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