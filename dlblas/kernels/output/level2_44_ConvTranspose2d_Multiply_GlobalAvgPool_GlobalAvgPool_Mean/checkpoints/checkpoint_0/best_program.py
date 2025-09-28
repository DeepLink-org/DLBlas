# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_reduction_kernel(
    x_ptr,
    output_ptr,
    multiplier,
    H, 
    W, 
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    spatial_size = H * W
    base = pid_b * (channels * spatial_size) + pid_c * spatial_size
    
    total = 0.0
    num_blocks = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, num_blocks):
        start = i * BLOCK_SIZE
        off = tl.arange(0, BLOCK_SIZE)
        idx = start + off
        mask = idx < spatial_size
        block_ptr = x_ptr + base + idx
        vals = tl.load(block_ptr, mask=mask, other=0.0)
        total += tl.sum(vals, axis=0)

    mean_val = total / spatial_size * multiplier
    output_index = pid_b * channels + pid_c
    tl.store(output_ptr + output_index, mean_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        batch, channels, H, W = x.shape
        x_contig = x.contiguous()
        output = torch.empty(batch, channels, 1, 1, device=x.device, dtype=x.dtype)
        output_flat = output.view(batch, channels)
        grid = (batch, channels)
        _fused_reduction_kernel[grid](x_contig, output_flat, self.multiplier, H, W, channels, BLOCK_SIZE=1024)
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]
# =================== EVOLVE-BLOCK-END ===================