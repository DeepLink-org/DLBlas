# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def global_avg_pool_clamp_kernel(
    input_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_d_stride,
    input_h_stride,
    input_w_stride,
    output_batch_stride,
    output_channel_stride,
    n_channels,
    depth,
    height,
    width,
    clamp_min,
    clamp_max,
    REDUCE_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // n_channels
    channel_id = pid % n_channels
    
    input_offset = (
        batch_id * input_batch_stride +
        channel_id * input_channel_stride
    )
    
    acc = 0.0
    num_elements = depth * height * width
    
    for idx in range(0, num_elements, REDUCE_SIZE):
        offsets = idx + tl.arange(0, REDUCE_SIZE)
        valid_mask = offsets < num_elements
        d = offsets // (height * width)
        hw = offsets % (height * width)
        h = hw // width
        w = hw % width
        
        d_offset = tl.where(valid_mask, d, 0)
        h_offset = tl.where(valid_mask, h, 0)
        w_offset = tl.where(valid_mask, w, 0)
        
        ptr = (
            input_ptr +
            d_offset * input_d_stride +
            h_offset * input_h_stride +
            w_offset * input_w_stride
        )
        values = tl.load(ptr + input_offset, mask=valid_mask, other=0.0)
        acc += tl.sum(values, axis=0)
    
    avg_val = acc / num_elements
    clamped = tl.minimum(tl.maximum(avg_val, clamp_min), clamp_max)
    
    output_offset = (
        batch_id * output_batch_stride +
        channel_id * output_channel_stride
    )
    tl.store(output_ptr + output_offset, clamped)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = x.contiguous()
        
        output = torch.empty(
            x.size(0), x.size(1), 1, 1, 1,
            device=x.device, dtype=x.dtype
        )
        
        grid = (x.size(0) * x.size(1),)
        global_avg_pool_clamp_kernel[grid](
            x,
            output,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            output.stride(0), output.stride(1),
            x.size(1),
            x.size(2), x.size(3), x.size(4),
            self.clamp_min,
            self.clamp_max,
            REDUCE_SIZE=128
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================