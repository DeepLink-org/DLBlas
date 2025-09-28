# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _depthwise_conv2d_kernel(
    x_ptr, weight_ptr, output_ptr,
    # Tensor dimensions
    batch_size, in_channels, input_h, input_w,
    kernel_h, kernel_w, output_h, output_w,
    # Strides for input tensor
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    # Strides for weight tensor
    w_stride_c, w_stride_h, w_stride_w,
    # Convolution parameters
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    # Block parameters
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Compute program indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)
    
    # Create block offsets
    offs_oh = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_ow = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for output boundaries
    mask_oh = offs_oh < output_h
    mask_ow = offs_ow < output_w
    output_mask = mask_oh[:, None] & mask_ow[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Compute input window positions
    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            # Calculate input positions with dilation
            ih = offs_oh * stride_h - padding_h + kh * dilation_h
            iw = offs_ow * stride_w - padding_w + kw * dilation_w
            
            # Create input mask
            input_mask = (ih >= 0) & (ih < input_h) & (iw >= 0) & (iw < input_w)
            full_mask = output_mask & input_mask
            
            # Compute memory offsets
            x_offsets = (pid_b * x_stride_b + 
                         pid_c * x_stride_c + 
                         ih[:, None] * x_stride_h + 
                         iw[None, :] * x_stride_w)
            
            # Load input block
            x_val = tl.load(x_ptr + x_offsets, mask=full_mask, other=0.0)
            
            # Load weight value
            w_val = tl.load(weight_ptr + pid_c * w_stride_c + kh * w_stride_h + kw * w_stride_w)
            
            # Accumulate
            acc += x_val * w_val
    
    # Compute output offsets
    output_offsets = (pid_b * output_h * output_w * in_channels +
                      pid_c * output_h * output_w +
                      offs_oh[:, None] * output_w +
                      offs_ow[None, :])
    
    # Store results
    tl.store(output_ptr + output_offsets, acc, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == in_channels, "Only depthwise convolution is supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(in_channels, 1, kernel_size_h, kernel_size_w)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous memory layout
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Compute output dimensions
        batch_size, _, input_h, input_w = x.shape
        output_h = (input_h + 2 * self.padding_h - 
                   self.dilation_h * (self.kernel_size_h - 1) - 1) // self.stride_h + 1
        output_w = (input_w + 2 * self.padding_w - 
                   self.dilation_w * (self.kernel_size_w - 1) - 1) // self.stride_w + 1
        
        # Create output tensor
        output = torch.empty(
            batch_size, self.out_channels, output_h, output_w,
            device=x.device, dtype=x.dtype
        )
        
        # Get tensor strides
        x_stride = x.stride()
        w_stride = weight.stride()
        
        # Configure kernel launch
        BLOCK_H, BLOCK_W = 16, 16
        grid = (
            batch_size, 
            self.in_channels,
            triton.cdiv(output_h, BLOCK_H),
            triton.cdiv(output_w, BLOCK_W)
        )
        
        # Launch kernel
        _depthwise_conv2d_kernel[grid](
            x, weight, output,
            batch_size, self.in_channels, input_h, input_w,
            self.kernel_size_h, self.kernel_size_w, output_h, output_w,
            x_stride[0], x_stride[1], x_stride[2], x_stride[3],
            w_stride[0], w_stride[2], w_stride[3],
            self.stride_h, self.stride_w, 
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]
# =================== EVOLVE-BLOCK-END ===================