# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.weight.contiguous()
        batch_size, in_channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        
        # Compute output dimensions
        out_height = (height - 1) * stride_h - 2 * padding_h + kernel_h
        out_width = (width - 1) * stride_w - 2 * padding_w + kernel_w
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width, 
                             device=x.device, dtype=x.dtype)
        
        # Calculate grid size for Triton
        def grid(META):
            grid_size = (triton.cdiv(out_height, META['BLOCK_H']) * 
                         triton.cdiv(out_width, META['BLOCK_W']) * 
                         batch_size * self.out_channels)
            return (grid_size,)
        
        # Call Triton kernel
        _conv_transpose2d_kernel[grid](
            x, weight, output, self.bias,
            batch_size, in_channels, self.out_channels,
            height, width, out_height, out_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_H=16, BLOCK_W=16
        )
        return output

@triton.jit
def _conv_transpose2d_kernel(
    x_ptr, weight_ptr, output_ptr, bias_ptr,
    batch_size, in_channels, out_channels,
    height, width, out_height, out_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
    weight_in_channel_stride, weight_out_channel_stride, weight_kernel_h_stride, weight_kernel_w_stride,
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks_w = tl.cdiv(out_width, BLOCK_W)
    num_blocks_h = tl.cdiv(out_height, BLOCK_H)
    num_blocks = num_blocks_h * num_blocks_w
    
    # Compute indices
    channel_idx = pid % out_channels
    batch_idx = (pid // out_channels) % batch_size
    block_idx = pid // (out_channels * batch_size)
    
    block_h_idx = (block_idx // num_blocks_w) * BLOCK_H
    block_w_idx = (block_idx % num_blocks_w) * BLOCK_W
    
    # Offsets for current block
    h_offsets = block_h_idx + tl.arange(0, BLOCK_H)
    w_offsets = block_w_idx + tl.arange(0, BLOCK_W)
    h_mask = h_offsets < out_height
    w_mask = w_offsets < out_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(in_channels):
        # Loop over kernel dimensions
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute corresponding input position
                input_h = (h_offsets[:, None] - kh + padding_h) / stride_h
                input_w = (w_offsets[None, :] - kw + padding_w) / stride_w
                
                # Check if input position is integer and within bounds
                is_int_h = (h_offsets[:, None] - kh + padding_h) % stride_h == 0
                is_int_w = (w_offsets[None, :] - kw + padding_w) % stride_w == 0
                valid_mask = is_int_h & is_int_w
                
                input_h_int = tl.math.floor(input_h).to(tl.int32)
                input_w_int = tl.math.floor(input_w).to(tl.int32)
                in_bounds = (input_h_int >= 0) & (input_h_int < height) & \
                            (input_w_int >= 0) & (input_w_int < width)
                
                # Combined mask
                mask = valid_mask & in_bounds & h_mask[:, None] & w_mask[None, :]
                
                # Compute memory offsets
                x_offset = (batch_idx * x_batch_stride + 
                            c * x_channel_stride + 
                            input_h_int * x_height_stride + 
                            input_w_int * x_width_stride)
                weight_offset = (c * weight_in_channel_stride + 
                                 channel_idx * weight_out_channel_stride + 
                                 kh * weight_kernel_h_stride + 
                                 kw * weight_kernel_w_stride)
                
                # Load input and weight
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                w_val = tl.load(weight_ptr + weight_offset)
                
                # Accumulate
                acc += tl.where(mask, x_val * w_val, 0.0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
    
    # Store results
    output_offset = (batch_idx * out_batch_stride + 
                     channel_idx * out_channel_stride + 
                     h_offsets[:, None] * out_height_stride + 
                     w_offsets[None, :] * out_width_stride)
    tl.store(output_ptr + output_offset, acc, mask=h_mask[:, None] & w_mask[None, :])

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
padding = (1, 2)

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================