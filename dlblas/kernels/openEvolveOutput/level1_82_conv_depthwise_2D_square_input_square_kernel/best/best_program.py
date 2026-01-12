# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    h_out,
    w_out,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    weight_stride_c, weight_stride_h, weight_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    
    n = pid0 // in_channels
    c = pid0 % in_channels
    i = pid1
    j = pid2
    
    acc = 0.0
    i0 = i * stride - padding
    j0 = j * stride - padding
    
    for di in range(kernel_size):
        for dj in range(kernel_size):
            h_idx = i0 + di
            w_idx = j0 + dj
            within_bounds = (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)
            x_offset = n * x_stride_n + c * x_stride_c + h_idx * x_stride_h + w_idx * x_stride_w
            x_val = tl.load(x_ptr + x_offset, mask=within_bounds, other=0.0)
            w_offset = c * weight_stride_c + di * weight_stride_h + dj * weight_stride_w
            w_val = tl.load(weight_ptr + w_offset)
            acc += x_val * w_val
    
    if has_bias:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val
        
    out_offset = n * output_stride_n + c * output_stride_c + i * output_stride_h + j * output_stride_w
    tl.store(output_ptr + out_offset, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        kernel_size = self.conv2d.kernel_size[0]
        stride = self.stride
        padding = self.padding
        
        h_out = (height + 2 * padding - kernel_size) // stride + 1
        w_out = (width + 2 * padding - kernel_size) // stride + 1
        
        output = torch.empty((batch_size, in_channels, h_out, w_out), device=x.device, dtype=x.dtype)
        weight = self.conv2d.weight.squeeze(1)
        bias = self.conv2d.bias
        has_bias = bias is not None
        
        grid = (batch_size * in_channels, h_out, w_out)
        depthwise_conv_kernel[grid](
            x, weight, bias, output,
            in_channels, height, width, kernel_size, stride, padding, h_out, w_out,
            has_bias,
            BLOCK_SIZE=1,
            x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
            weight_stride_c=weight.stride(0), weight_stride_h=weight.stride(1), weight_stride_w=weight.stride(2),
            output_stride_n=output.stride(0), output_stride_c=output.stride(1), output_stride_h=output.stride(2), output_stride_w=output.stride(3),
        )
        return output

# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================