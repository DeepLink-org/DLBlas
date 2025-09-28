# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

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
        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            self.out_channels_per_group, 
            kernel_size, kernel_size, kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth_in, height_in, width_in = x.shape
        depth_out = (depth_in - 1) * self.stride - 2 * self.padding + self.kernel_size
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.kernel_size
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        output = torch.zeros(
            batch_size, self.out_channels, depth_out, height_out, width_out,
            device=x.device, dtype=x.dtype
        )
        
        total_elements = (
            batch_size * 
            self.groups * 
            self.out_channels_per_group * 
            depth_out * 
            height_out * 
            width_out
        )
        
        if total_elements > 0:
            _conv_transpose3d_forward_kernel[ (total_elements,) ](
                x, output, self.weight,
                batch_size, self.groups, 
                self.in_channels_per_group, self.out_channels_per_group,
                depth_in, height_in, width_in,
                depth_out, height_out, width_out,
                self.stride, self.padding, self.kernel_size,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
                self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), 
                self.weight.stride(3), self.weight.stride(4),
                total_elements
            )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output

@triton.jit
def _conv_transpose3d_forward_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    batch_size, groups, in_channels_per_group, out_channels_per_group,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    stride, padding, kernel_size,
    x_batch_stride, x_in_channel_stride, x_d_stride, x_h_stride, x_w_stride,
    out_batch_stride, out_channel_stride, out_d_stride, out_h_stride, out_w_stride,
    weight_in_channel_stride, weight_out_channel_stride, 
    weight_d_stride, weight_h_stride, weight_w_stride,
    total_elements,
):
    pid = tl.program_id(0)
    if pid >= total_elements:
        return
    
    # Decompose linear index into tensor indices
    width_out = width_out
    height_out = height_out
    depth_out = depth_out
    channels_per_group = out_channels_per_group
    
    w_out = pid % width_out
    pid = pid // width_out
    h_out = pid % height_out
    pid = pid // height_out
    d_out = pid % depth_out
    pid = pid // depth_out
    c_out = pid % channels_per_group
    pid = pid // channels_per_group
    group_idx = pid % groups
    batch_idx = pid // groups
    
    # Initialize accumulator
    acc = 0.0
    
    # Global input channel index
    input_channel_start = group_idx * in_channels_per_group
    input_channel_end = input_channel_start + in_channels_per_group
    
    # Iterate through kernel positions
    for kd in range(kernel_size):
        d_in_val = d_out + padding - kd
        if d_in_val % stride == 0:
            d_in = d_in_val // stride
            if d_in >= 0 and d_in < depth_in:
                for kh in range(kernel_size):
                    h_in_val = h_out + padding - kh
                    if h_in_val % stride == 0:
                        h_in = h_in_val // stride
                        if h_in >= 0 and h_in < height_in:
                            for kw in range(kernel_size):
                                w_in_val = w_out + padding - kw
                                if w_in_val % stride == 0:
                                    w_in = w_in_val // stride
                                    if w_in >= 0 and w_in < width_in:
                                        # Iterate through input channels in group
                                        for c_in in range(input_channel_start, input_channel_end):
                                            # Load input value
                                            x_offset = (
                                                batch_idx * x_batch_stride +
                                                c_in * x_in_channel_stride +
                                                d_in * x_d_stride +
                                                h_in * x_h_stride +
                                                w_in * x_w_stride
                                            )
                                            input_val = tl.load(x_ptr + x_offset)
                                            
                                            # Load weight value
                                            local_c_in = c_in - input_channel_start
                                            weight_offset = (
                                                c_in * weight_in_channel_stride +
                                                c_out * weight_out_channel_stride +
                                                kd * weight_d_stride +
                                                kh * weight_h_stride +
                                                kw * weight_w_stride
                                            )
                                            weight_val = tl.load(weight_ptr + weight_offset)
                                            
                                            # Accumulate
                                            acc += input_val * weight_val
    
    # Compute output index and store result
    output_channel = group_idx * out_channels_per_group + c_out
    out_offset = (
        batch_idx * out_batch_stride +
        output_channel * out_channel_stride +
        d_out * out_d_stride +
        h_out * out_h_stride +
        w_out * out_w_stride
    )
    tl.store(output_ptr + out_offset, acc)

# Test code
import math
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 3
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
# =================== EVOLVE-BLOCK-END ===================