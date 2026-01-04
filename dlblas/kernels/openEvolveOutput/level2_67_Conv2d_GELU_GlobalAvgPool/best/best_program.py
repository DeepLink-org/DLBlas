# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def global_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    height,
    width,
    num_elements,
    output_batch_stride,
    output_channel_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    base_ptr = input_ptr + pid_batch * input_batch_stride + pid_channel * input_channel_stride
    total = 0.0
    
    for i in range(0, num_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        h = offsets // width
        w = offsets % width
        ptrs = base_ptr + h * input_height_stride + w * input_width_stride
        data = tl.load(ptrs, mask=mask, other=0.0)
        total += tl.sum(data, axis=0)
    
    avg = total / num_elements
    output_offset = pid_batch * output_batch_stride + pid_channel * output_channel_stride
    tl.store(output_ptr + output_offset, avg)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.gelu(x)
        
        batch, channels, h, w = x.shape
        num_elements = h * w
        output = torch.empty((batch, channels), device=x.device, dtype=x.dtype)
        
        batch_stride, channel_stride, height_stride, width_stride = x.stride()
        output_batch_stride, output_channel_stride = output.stride()
        
        grid = (batch, channels)
        global_avg_pool2d_kernel[grid](
            x, output,
            batch_stride, channel_stride, height_stride, width_stride,
            h, w, num_elements,
            output_batch_stride, output_channel_stride,
            BLOCK_SIZE=1024
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