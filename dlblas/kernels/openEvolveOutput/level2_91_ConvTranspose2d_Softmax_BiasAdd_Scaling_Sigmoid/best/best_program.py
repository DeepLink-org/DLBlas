# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    input_ptr, bias_ptr, output_ptr, scaling_factor,
    batch_size, channels, height, width,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    bias_channel_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    if pid_b >= batch_size or pid_h >= height or pid_w >= width:
        return
        
    base_input = pid_b * input_batch_stride + pid_h * input_height_stride + pid_w * input_width_stride
    base_output = pid_b * output_batch_stride + pid_h * output_height_stride + pid_w * output_width_stride
    
    offs_c = tl.arange(0, BLOCK_SIZE_C)
    input_ptrs = input_ptr + base_input + offs_c * input_channel_stride
    output_ptrs = output_ptr + base_output + offs_c * output_channel_stride
    
    vec = tl.load(input_ptrs, mask=offs_c < channels, other=float('-inf'))
    
    max_val = tl.max(vec, axis=0)
    vec = tl.exp(vec - max_val)
    sum_val = tl.sum(vec, axis=0)
    softmax_out = vec / sum_val
    
    bias_vec = tl.load(bias_ptr + offs_c * bias_channel_stride, mask=offs_c < channels, other=0.0)
    biased = softmax_out + bias_vec
    scaled = biased * scaling_factor
    result = tl.sigmoid(scaled)
    
    tl.store(output_ptrs, result, mask=offs_c < channels)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape).view(-1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        batch_size, channels, height, width = x.shape
        output = torch.empty_like(x)
        grid = (batch_size, height, width)
        
        fused_softmax_bias_scale_sigmoid_kernel[grid](
            x, self.bias, output, self.scaling_factor,
            batch_size, channels, height, width,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.bias.stride(0),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE_C=64
        )
        return output

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================