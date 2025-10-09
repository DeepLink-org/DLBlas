# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    @staticmethod
    @triton.jit
    def _depthwise_conv_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        in_channels, height, width,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        kernel_size,
        output_height, output_width,
        x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
        weight_channel_stride, weight_height_stride, weight_width_stride,
        output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
        BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_hw = tl.program_id(2)
        
        output_width_blocks = tl.cdiv(output_width, BLOCK_W)
        pid_h = pid_hw // output_width_blocks
        pid_w = pid_hw % output_width_blocks
        
        h_start = pid_h * BLOCK_H
        w_start = pid_w * BLOCK_W
        
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        w_offsets = w_start + tl.arange(0, BLOCK_W)
        
        h_mask = h_offsets < output_height
        w_mask = w_offsets < output_width
        block_mask = h_mask[:, None] & w_mask[None, :]
        
        accumulator = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        base_ptr = x_ptr + pid_b * x_batch_stride + pid_c * x_channel_stride
        
        for ki in range(kernel_size):
            for kj in range(kernel_size):
                input_i = h_offsets * stride_h + ki * dilation_h - padding_h
                input_j = w_offsets * stride_w + kj * dilation_w - padding_w
                
                input_i_mask = (input_i >= 0) & (input_i < height)
                input_j_mask = (input_j >= 0) & (input_j < width)
                input_mask = input_i_mask[:, None] & input_j_mask[None, :] & block_mask
                
                input_val = tl.load(
                    base_ptr + input_i[:, None] * x_height_stride + input_j[None, :] * x_width_stride,
                    mask=input_mask,
                    other=0.0
                )
                
                weight_val = tl.load(
                    weight_ptr + pid_c * weight_channel_stride + ki * weight_height_stride + kj * weight_width_stride
                )
                
                accumulator += input_val * weight_val
        
        output_ptr_base = output_ptr + pid_b * output_batch_stride + pid_c * output_channel_stride
        output_offsets = h_offsets[:, None] * output_height_stride + w_offsets[None, :] * output_width_stride
        tl.store(output_ptr_base + output_offsets, accumulator, mask=block_mask)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = self.depthwise.stride
        padding = self.depthwise.padding
        dilation = self.depthwise.dilation
        weight = self.depthwise.weight
        bias = self.depthwise.bias
        
        batch, in_channels, height, width = x.shape
        kernel_size = weight.shape[2]
        
        output_height = (height + 2 * padding[0] - dilation[0] * (kernel_size - 1) - 1) // stride[0] + 1
        output_width = (width + 2 * padding[1] - dilation[1] * (kernel_size - 1) - 1) // stride[1] + 1
        
        x = x.contiguous()
        weight = weight.contiguous()
        output = torch.empty((batch, in_channels, output_height, output_width), device=x.device, dtype=x.dtype)
        
        grid = (batch, in_channels, triton.cdiv(output_height, 16) * triton.cdiv(output_width, 16))
        
        self._depthwise_conv_kernel[grid](
            x, weight, output,
            in_channels, height, width,
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            kernel_size,
            output_height, output_width,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(2), weight.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_H=16, BLOCK_W=16
        )
        
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)
            
        output = self.pointwise(output)
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================