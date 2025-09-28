# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    total_elements = batch_size * out_channels * out_h * out_w
    elements_per_program = tl.cdiv(total_elements, num_pid)
    
    start_idx = pid * elements_per_program
    end_idx = min(start_idx + elements_per_program, total_elements)
    
    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups
    
    for idx in range(start_idx, end_idx):
        # Compute output indices
        tmp = idx
        b = tmp // (out_channels * out_h * out_w)
        tmp = tmp % (out_channels * out_h * out_w)
        c_out = tmp // (out_h * out_w)
        tmp = tmp % (out_h * out_w)
        i = tmp // out_w
        j = tmp % out_w
        
        group_idx = c_out // out_channels_per_group
        c_out_group = c_out % out_channels_per_group
        start_c_in = group_idx * in_channels_per_group
        
        total = 0.0
        for di in range(kernel_h):
            for dj in range(kernel_w):
                in_i = i + padding_h - di * dilation_h
                in_j = j + padding_w - dj * dilation_w
                
                if (in_i % stride_h == 0) & (in_j % stride_w == 0):
                    in_i_idx = in_i // stride_h
                    in_j_idx = in_j // stride_w
                    
                    if (in_i_idx >= 0) & (in_i_idx < in_h) & (in_j_idx >= 0) & (in_j_idx < in_w):
                        for c_in_offset in range(0, in_channels_per_group, BLOCK_C):
                            c_in = start_c_in + c_in_offset
                            mask = c_in_offset + tl.arange(0, BLOCK_C) < in_channels_per_group
                            
                            # Calculate input and weight pointers
                            x_offset = (b * in_channels + c_in) * in_h * in_w + in_i_idx * in_w + in_j_idx
                            w_offset_base = (c_in * out_channels_per_group + c_out_group) * kernel_h * kernel_w + di * kernel_w + dj
                            w_offset_step = out_channels_per_group * kernel_h * kernel_w
                            
                            # Vectorized loads
                            input_vals = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_C) * in_h * in_w, mask=mask, other=0.0)
                            weight_offsets = w_offset_base + tl.arange(0, BLOCK_C) * w_offset_step
                            weight_vals = tl.load(w_ptr + weight_offsets, mask=mask, other=0.0)
                            
                            # Accumulate
                            total += tl.sum(input_vals * weight_vals)
        
        # Store final result
        output_offset = b * out_channels * out_h * out_w + c_out * out_h * out_w + i * out_w + j
        tl.store(output_ptr + output_offset, total)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels // groups, kernel_size[0], kernel_size[1])))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_channels, in_h, in_w = x.shape
        assert in_channels == self.in_channels, "Input channels mismatch"
        
        # Calculate output dimensions
        out_h = (in_h - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + 1
        out_w = (in_w - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + 1
        
        output = torch.zeros((batch, self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
        
        # Launch kernel
        total_elements = batch * self.out_channels * out_h * out_w
        grid = lambda meta: (triton.cdiv(total_elements, 256),)
        
        conv_transpose2d_kernel[grid](
            x,
            self.weight,
            output,
            batch,
            self.in_channels,
            in_h,
            in_w,
            self.out_channels,
            out_h,
            out_w,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            BLOCK_C=64,
        )
        
        if self.bias is not None:
            output += self.bias[None, :, None, None]
            
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]
# =================== EVOLVE-BLOCK-END ===================