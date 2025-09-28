# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    # Input tensor pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Tensor dimensions
    input_batch, input_channels, input_depth, input_height, input_width,
    output_channels, output_depth, output_height, output_width,
    # Kernel parameters
    kernel_size, stride, padding, groups,
    # Memory strides
    input_batch_stride, input_channel_stride, input_d_stride, input_h_stride, input_w_stride,
    weight_in_channel_stride, weight_out_channel_stride, weight_d_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_d_stride, output_h_stride, output_w_stride,
    # Blocking parameters
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_HW: tl.constexpr
):
    # Program ID mapping
    pid_bc = tl.program_id(0)  # Batch*channel groups
    pid_d = tl.program_id(1)   # Output depth
    pid_hw = tl.program_id(2)  # Combined height*width

    # Reconstruct indices
    batch_idx = pid_bc // (output_channels // groups)
    group_idx = pid_bc % (output_channels // groups)
    c_out = group_idx + (groups * (pid_bc // (output_channels // groups)) % groups) * (output_channels // groups)
    
    # Extract spatial coordinates
    hw_idx = pid_hw
    h_idx = hw_idx // output_width
    w_idx = hw_idx % output_width
    d_idx = pid_d

    # Calculate input channel range
    group_channels = input_channels // groups
    c_in_start = (c_out // (output_channels // groups)) * group_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input indices
                d_in_val = d_idx + padding - kd
                h_in_val = h_idx + padding - kh
                w_in_val = w_idx + padding - kw
                
                # Check if indices are valid
                valid_d = (d_in_val >= 0) & (d_in_val < input_depth * stride) & (d_in_val % stride == 0)
                valid_h = (h_in_val >= 0) & (h_in_val < input_height * stride) & (h_in_val % stride == 0)
                valid_w = (w_in_val >= 0) & (w_in_val < input_width * stride) & (w_in_val % stride == 0)
                
                if valid_d & valid_h & valid_w:
                    d_in = d_in_val // stride
                    h_in = h_in_val // stride
                    w_in = w_in_val // stride
                    
                    # Load input values in batches
                    for c_block in range(0, group_channels, BLOCK_C):
                        c_off = tl.arange(0, BLOCK_C)
                        c_in = c_in_start + c_block + c_off
                        
                        # Check channel boundaries
                        mask = c_off < (group_channels - c_block)
                        
                        # Input offset calculation
                        input_offset = (
                            (batch_idx * input_batch_stride) +
                            (c_in * input_channel_stride) +
                            (d_in * input_d_stride) +
                            (h_in * input_h_stride) +
                            (w_in * input_w_stride)
                        )
                        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                        
                        # Weight offset calculation
                        weight_offset = (
                            (c_in * weight_in_channel_stride) +
                            (group_idx * weight_out_channel_stride) +
                            (kd * weight_d_stride) +
                            (kh * weight_h_stride) +
                            (kw * weight_w_stride)
                        )
                        weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c_out)
        acc += bias
    
    # Output offset calculation
    output_offset = (
        (batch_idx * output_batch_stride) +
        (c_out * output_channel_stride) +
        (d_idx * output_d_stride) +
        (h_idx * output_h_stride) +
        (w_idx * output_w_stride)
    )
    
    # Store result
    tl.store(output_ptr + output_offset, acc.sum())

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
        
        # Weight initialization
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size, kernel_size, kernel_size
        ))
        
        # Bias initialization
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions
        def calc_output_size(dim):
            return (dim - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        batch_size, _, in_depth, in_height, in_width = x.shape
        out_depth = calc_output_size(in_depth)
        out_height = calc_output_size(in_height)
        out_width = calc_output_size(in_width)
        
        # Create output tensor
        output = torch.empty(
            batch_size, self.out_channels, out_depth, out_height, out_width,
            device=x.device, dtype=x.dtype
        )
        
        # Get tensor strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = output.stride()
        
        # Kernel configuration
        BLOCK_C = 16
        grid = (
            batch_size * self.groups,
            out_depth,
            out_height * out_width
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, self.in_channels, in_depth, in_height, in_width,
            self.out_channels, out_depth, out_height, out_width,
            self.kernel_size, self.stride, self.padding, self.groups,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
            BLOCK_C, 1, 1
        )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================