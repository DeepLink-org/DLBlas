# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def subtract_mean_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    base = pid_batch * n_channels * spatial_size + pid_channel * spatial_size
    total = 0.0
    
    # Efficient reduction using vectorized operations
    for offset in range(0, spatial_size, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < spatial_size
        block = tl.load(input_ptr + base + block_offsets, mask=mask, other=0.0)
        total += tl.sum(block, axis=0)
    
    mean = total / spatial_size
    
    # Vectorized subtraction with fused load/store
    for offset in range(0, spatial_size, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < spatial_size
        block = tl.load(input_ptr + base + block_offsets, mask=mask, other=0.0)
        result = block - mean
        tl.store(output_ptr + base + block_offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        
        # Optimized mean subtraction with Triton kernel
        if not x.is_contiguous():
            x = x.contiguous()
        output = torch.empty_like(x)
        batch_size, out_channels, D, H, W = x.shape
        spatial_size = D * H * W
        grid = (batch_size, out_channels)
        
        subtract_mean_kernel[grid](
            x, output, out_channels, spatial_size, 
            BLOCK_SIZE=min(1024, spatial_size)
        )
        return output

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================