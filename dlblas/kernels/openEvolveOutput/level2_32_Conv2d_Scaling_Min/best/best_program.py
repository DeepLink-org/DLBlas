# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _min_reduction_kernel(
    x_ptr,
    output_ptr,
    spatial_size,
    x_batch_stride,
    x_channel_stride,
    x_spatial_stride,
    output_batch_stride,
    output_spatial_stride,
    channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_spatial_block = tl.program_id(1)
    
    spatial_start = pid_spatial_block * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < spatial_size
    
    base_batch = pid_batch * x_batch_stride
    ptr0 = x_ptr + base_batch + 0 * x_channel_stride + spatial_offsets
    min_values = tl.load(ptr0, mask=spatial_mask, other=float('inf'))
    
    for c in range(1, channels):
        ptr = x_ptr + base_batch + c * x_channel_stride + spatial_offsets
        vals = tl.load(ptr, mask=spatial_mask, other=float('inf'))
        min_values = tl.minimum(min_values, vals)
    
    out_ptr = output_ptr + pid_batch * output_batch_stride + spatial_offsets
    tl.store(out_ptr, min_values, mask=spatial_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = x * self.scale_factor
        
        batch_size, channels, height, width = x.shape
        spatial_size = height * width
        x_flat = x.contiguous().view(batch_size, channels, spatial_size)
        output = torch.empty(batch_size, 1, spatial_size, device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE = 128
        grid = (batch_size, triton.cdiv(spatial_size, BLOCK_SIZE))
        
        _min_reduction_kernel[grid](
            x_flat, output,
            spatial_size,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            output.stride(0),
            output.stride(2),
            channels,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output.view(batch_size, 1, height, width)

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
# =================== EVOLVE-BLOCK-END ===================