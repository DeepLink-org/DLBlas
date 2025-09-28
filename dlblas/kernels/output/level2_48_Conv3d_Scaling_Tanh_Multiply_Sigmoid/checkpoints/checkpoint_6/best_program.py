# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['spatial_size'],
)
@triton.jit
def _fused_pointwise_kernel(
    input_ptr,
    scaling_ptr,
    bias_ptr,
    output_ptr,
    spatial_size,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    spatial_offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < spatial_size
    
    full_offsets = pid_bc * spatial_size + spatial_offsets
    c = pid_bc % out_channels
    
    x = tl.load(input_ptr + full_offsets, mask=mask)
    s = tl.load(scaling_ptr + c)
    b = tl.load(bias_ptr + c)
    
    scaled = x * s
    activated = tl.libdevice.tanh(scaled)
    biased = activated * b
    output = tl.sigmoid(biased)
    
    tl.store(output_ptr + full_offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x).contiguous()
        output = torch.empty_like(x)
        
        B, C, D, H, W = x.shape
        spatial_size = D * H * W
        total_bc = B * C
        
        if x.numel() > 0:
            grid = (total_bc, triton.cdiv(spatial_size, BLOCK_SIZE))
            _fused_pointwise_kernel[grid](
                x, 
                self.scaling_factor.view(-1), 
                self.bias.view(-1), 
                output,
                spatial_size,
                C,
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