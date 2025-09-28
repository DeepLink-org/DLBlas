# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, height, width,
    out_channels, kernel_size,
    output_height, output_width,
    stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_output_elements = batch_size * out_channels * output_height * output_width
    if pid >= total_output_elements:
        return

    b = pid // (out_channels * output_height * output_width)
    pid_rest = pid % (out_channels * output_height * output_width)
    oc = pid_rest // (output_height * output_width)
    pid_rest2 = pid_rest % (output_height * output_width)
    y = pid_rest2 // output_width
    x = pid_rest2 % output_width

    acc = 0.0
    for ic in range(in_channels):
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                iy = y * stride + ky * dilation - padding
                ix = x * stride + kx * dilation - padding
                within_bounds = (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
                input_offset = b * (in_channels * height * width) + ic * (height * width) + iy * width + ix
                weight_offset = oc * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + ky * kernel_size + kx
                input_val = tl.load(input_ptr + input_offset, mask=within_bounds, other=0.0)
                weight_val = tl.load(weight_ptr + weight_offset)
                acc += input_val * weight_val

    bias_val = tl.load(bias_ptr + oc)
    acc += bias_val
    output_offset = b * (out_channels * output_height * output_width) + oc * (output_height * output_width) + y * output_width + x
    tl.store(output_ptr + output_offset, acc)

def custom_conv2d(x, weight, bias, stride=1, padding=0, dilation=1):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=x.dtype)
    total_output_elements = batch_size * out_channels * output_height * output_width

    grid = lambda meta: (triton.cdiv(total_output_elements, meta['BLOCK_SIZE']),)
    conv2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, height, width,
        out_channels, kernel_size,
        output_height, output_width,
        stride, padding, dilation,
        BLOCK_SIZE=1024
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = custom_conv2d(x, self.weight, self.bias)
        x = self.bn(x)
        x = x * self.scaling_factor
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================