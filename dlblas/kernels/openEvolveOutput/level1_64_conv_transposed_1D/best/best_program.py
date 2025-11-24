# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose1d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, in_length,
    out_channels, out_length, kernel_size,
    stride, padding, groups,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t_block = tl.program_id(2)
    
    t_start = pid_t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < out_length

    out_channels_per_group = out_channels // groups
    group_idx = pid_c // out_channels_per_group
    c_out_index = pid_c % out_channels_per_group
    in_channels_per_group = in_channels // groups
    c_in_start = group_idx * in_channels_per_group

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    
    for k in range(kernel_size):
        num = t_offsets + padding - k
        divisible = num % stride == 0
        t_in = num // stride
        in_bounds = (t_in >= 0) & (t_in < in_length)
        valid = divisible & in_bounds & t_mask

        num_ic_blocks = tl.cdiv(in_channels_per_group, BLOCK_C)
        for ic_block in range(num_ic_blocks):
            ic_start = ic_block * BLOCK_C
            ic_offsets = ic_start + tl.arange(0, BLOCK_C)
            ic_mask = ic_offsets < in_channels_per_group
            c_in_indices = c_in_start + ic_offsets

            weight_ptrs = weight_ptr + c_in_indices * (out_channels_per_group * kernel_size) + c_out_index * kernel_size + k
            weight_vals = tl.load(weight_ptrs, mask=ic_mask, other=0.0)

            input_ptrs = input_ptr + pid_b * (in_channels * in_length) + (c_in_indices[:, None]) * in_length + t_in[None, :]
            input_vals = tl.load(input_ptrs, mask=ic_mask[:, None] & valid[None, :], other=0.0)
            
            products = input_vals * weight_vals[:, None]
            partial_sum = tl.sum(products, axis=0)
            acc = tl.where(valid, acc + partial_sum, acc)

    output_ptrs = output_ptr + pid_b * (out_channels * out_length) + pid_c * out_length + t_offsets
    tl.store(output_ptrs, acc, mask=t_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_length = x.shape
        assert in_channels == self.in_channels
        
        out_length = (in_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output = torch.zeros((batch_size, self.out_channels, out_length), device=x.device, dtype=x.dtype)
        
        BLOCK_C = min(64, self.in_channels // self.groups)
        BLOCK_T = 128
        grid = (batch_size, self.out_channels, triton.cdiv(out_length, BLOCK_T))
        
        conv_transpose1d_kernel[grid](
            x, self.weight, output,
            batch_size, in_channels, in_length,
            self.out_channels, out_length, self.kernel_size,
            self.stride, self.padding, self.groups,
            BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T
        )
        
        if self.bias is not None:
            output += self.bias[:, None]
            
        return output

# Test code
batch_size = 16
in_channels = 64
out_channels = 3
kernel_size = 3
length = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================