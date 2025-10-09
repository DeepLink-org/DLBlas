# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _min_tanh_tanh_kernel(
    input_ptr,
    output_ptr,
    channels,
    height,
    width,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    base = pid_b * stride_b + pid_h * stride_h + pid_w * stride_w
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < channels
    channel_ptrs = input_ptr + base + c_offsets * stride_c
    channel_vals = tl.load(channel_ptrs, mask=c_mask, other=float('inf'))
    
    min_val = tl.min(channel_vals, axis=0)
    out_val = tl.tanh(tl.tanh(min_val))
    
    output_offset = pid_b * (height * width) + pid_h * width + pid_w
    tl.store(output_ptr + output_offset, out_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    def forward(self, x):
        x = self.conv(x)
        batch, channels, height, width = x.shape
        x_cont = x.contiguous()
        output = torch.empty(batch, 1, height, width, device=x.device, dtype=x.dtype)
        
        grid = (batch, height, width)
        stride_b = x_cont.stride(0)
        stride_c = x_cont.stride(1)
        stride_h = x_cont.stride(2)
        stride_w = x_cont.stride(3)
        BLOCK_SIZE_C = 2 ** math.ceil(math.log2(channels))
        
        _min_tanh_tanh_kernel[grid](
            x_cont, output,
            channels, height, width,
            stride_b, stride_c, stride_h, stride_w,
            BLOCK_SIZE_C
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================