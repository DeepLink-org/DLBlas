# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _conv_transpose2d_forward(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride,
    padding,
    groups,
    in_channels,
    out_channels,
    kernel_size,
    height,
    width,
    height_out,
    width_out,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_kernel_h_stride,
    weight_kernel_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Compute spatial indices
    pid_h = pid_hw // tl.cdiv(width_out, BLOCK_SIZE_W)
    pid_w = pid_hw % tl.cdiv(width_out, BLOCK_SIZE_W)
    
    # Create ranges for block processing
    c_offsets = pid_c_out * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    h_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for boundary checks
    c_mask = c_offsets < out_channels
    h_mask = h_offsets < height_out
    w_mask = w_offsets < width_out
    spatial_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize output block
    output_block = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Calculate group information
    channels_per_group = out_channels // groups
    group_id = c_offsets // channels_per_group
    
    # Process input channels
    for c_in in range(in_channels // groups):
        # Precompute input channel index
        c_in_idx = group_id * (in_channels // groups) + c_in
        
        # Process kernel elements
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input positions
                h_in = (h_offsets + padding - kh) // stride
                w_in = (w_offsets + padding - kw) // stride
                
                # Create valid position mask
                valid_mask = spatial_mask
                valid_mask &= (h_offsets + padding - kh) % stride == 0
                valid_mask &= (w_offsets + padding - kw) % stride == 0
                valid_mask &= (h_in >= 0) & (h_in < height)
                valid_mask &= (w_in >= 0) & (w_in < width)
                
                # Load input values
                x_ptrs = (
                    x_ptr + 
                    pid_b * x_batch_stride +
                    c_in_idx[:, None, None] * x_channel_stride +
                    h_in[None, :, None] * x_height_stride +
                    w_in[None, None, :] * x_width_stride
                )
                x_vals = tl.load(x_ptrs, mask=valid_mask, other=0.0)
                
                # Load weight values
                weight_ptrs = (
                    weight_ptr +
                    c_offsets[:, None, None] * weight_out_channel_stride +
                    c_in_idx[:, None, None] * weight_in_channel_stride +
                    kh * weight_kernel_h_stride +
                    kw * weight_kernel_w_stride
                )
                weight_vals = tl.load(weight_ptrs, mask=c_mask[:, None, None], other=0.0)
                
                # Accumulate results
                output_block += x_vals * weight_vals
    
    # Add bias if present
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + c_offsets
        bias_vals = tl.load(bias_ptrs, mask=c_mask, other=0.0)
        output_block += bias_vals[:, None, None]
    
    # Store output
    output_ptrs = (
        output_ptr +
        pid_b * output_batch_stride +
        c_offsets[:, None, None] * output_channel_stride +
        h_offsets[None, :, None] * output_height_stride +
        w_offsets[None, None, :] * output_width_stride
    )
    tl.store(output_ptrs, output_block, mask=spatial_mask & c_mask[:, None, None])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight and bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        batch_size, _, height, width = x.shape
        height_out = (height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        width_out = (width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        # Create output tensor
        output = torch.empty(
            (batch_size, self.out_channels, height_out, width_out),
            device=x.device, 
            dtype=x.dtype
        )
        
        # Configure Triton kernel grid
        BLOCK_SIZE_C = 16
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        
        grid = (
            batch_size, 
            triton.cdiv(self.out_channels, BLOCK_SIZE_C),
            triton.cdiv(height_out, BLOCK_SIZE_H) * triton.cdiv(width_out, BLOCK_SIZE_W)
        )
        
        # Get tensor strides
        x_stride = x.stride()
        weight_stride = self.weight.stride()
        output_stride = output.stride()
        
        # Launch kernel
        _conv_transpose2d_forward[grid](
            x,
            self.weight,
            self.bias if self.bias is not None else None,
            output,
            self.stride,
            self.padding,
            self.groups,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            height,
            width,
            height_out,
            width_out,
            x_stride[0], x_stride[1], x_stride[2], x_stride[3],
            weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
            output_stride[0], output_stride[1], output_stride[2], output_stride[3],
            BLOCK_SIZE_C,
            BLOCK_SIZE_H,
            BLOCK_SIZE_W
        )
        
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================