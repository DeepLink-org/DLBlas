# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _pointwise_ops_kernel(
    input_ptr,
    bias_ptr,
    scale_ptr,
    output_ptr,
    n_elements,
    C, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate channel indices
    spatial_size = H * W
    channel_idx = (offsets // spatial_size) % C
    
    # Load inputs
    x = tl.load(input_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + channel_idx, mask=mask)
    scale = tl.load(scale_ptr + channel_idx, mask=mask)
    
    # Compute fused operations
    x = (x + bias) * scale
    x = tl.sigmoid(x)
    
    # Store results
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        
        # Get tensor dimensions
        N, C, H, W = x.shape
        n_elements = N * C * H * W
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Reshape parameters to 1D for kernel access
        bias_1d = self.bias.view(C)
        scale_1d = self.scale.view(C)
        
        # Launch Triton kernel for fused operations
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _pointwise_ops_kernel[grid](
            x, bias_1d, scale_1d, output,
            n_elements, C, H, W,
            BLOCK_SIZE=1024
        )
        
        x = self.group_norm(output)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
# =================== EVOLVE-BLOCK-END ===================