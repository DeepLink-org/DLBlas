# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _conv_transpose3d_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    in_channels_per_group,
    out_channels_per_group,
    depth_in, height_in, width_in,
    depth_out, height_out, width_out,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    kernel_d, kernel_h, kernel_w,
    has_bias: tl.constexpr,
    BLOCK_SIZE_CH: tl.constexpr,
):
    pid_gbc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    groups = tl.num_programs(0) // (depth_out * height_out * width_out * BLOCK_SIZE_CH)
    group_id = pid_gbc // (depth_out * height_out * width_out * BLOCK_SIZE_CH)
    remainder = pid_gbc % (depth_out * height_out * width_out * BLOCK_SIZE_CH)
    block_ch = (remainder // (depth_out * height_out * width_out)) * BLOCK_SIZE_CH
    spatial_idx = remainder % (depth_out * height_out * width_out)
    
    batch_id = spatial_idx // (height_out * width_out)
    hw_idx = spatial_idx % (height_out * width_out)
    d_out = pid_d
    h_out = hw_idx // width_out
    w_out = hw_idx % width_out
    
    out_ch_offset = tl.program_id(0) % out_channels_per_group
    out_channel_id_in_group = block_ch + out_ch_offset
    
    if d_out >= depth_out or h_out >= height_out or w_out >= width_out:
        return
    if out_channel_id_in_group >= out_channels_per_group:
        return

    global_out_ch = group_id * out_channels_per_group + out_channel_id_in_group
    base_x_ptr = x_ptr + batch_id * in_channels_per_group * depth_in * height_in * width_in * groups
    base_x_ptr += group_id * in_channels_per_group * depth_in * height_in * width_in
    base_weight_ptr = weight_ptr + group_id * in_channels_per_group * out_channels_per_group * kernel_d * kernel_h * kernel_w
    base_weight_ptr += out_channel_id_in_group * in_channels_per_group * kernel_d * kernel_h * kernel_w

    accumulator = 0.0
    for kd in range(kernel_d):
        d_in = d_out * stride_d - padding_d + kd
        # Replace continue with conditional block
        if d_in >= 0 and d_in < depth_in:
            for kh in range(kernel_h):
                h_in = h_out * stride_h - padding_h + kh
                # Replace continue with conditional block
                if h_in >= 0 and h_in < height_in:
                    for kw in range(kernel_w):
                        w_in = w_out * stride_w - padding_w + kw
                        # Replace continue with conditional block
                        if w_in >= 0 and w_in < width_in:
                            spatial_offset = d_in * height_in * width_in + h_in * width_in + w_in
                            for ch_block in range(0, in_channels_per_group, BLOCK_SIZE_CH):
                                ch_offsets = ch_block + tl.arange(0, BLOCK_SIZE_CH)
                                mask = ch_offsets < in_channels_per_group
                                
                                x_vals = tl.load(
                                    base_x_ptr + ch_offsets * depth_in * height_in * width_in + spatial_offset,
                                    mask=mask,
                                    other=0.0
                                )
                                
                                weight_offset = (kd * kernel_h * kernel_w + kh * kernel_w + kw) * in_channels_per_group + ch_offsets
                                w_vals = tl.load(
                                    base_weight_ptr + weight_offset,
                                    mask=mask,
                                    other=0.0
                                )
                                
                                accumulator += tl.sum(x_vals * w_vals)

    if has_bias:
        bias_val = tl.load(bias_ptr + global_out_ch)
        accumulator += bias_val

    output_offset = (batch_id * groups * out_channels_per_group * depth_out * height_out * width_out +
                     global_out_ch * depth_out * height_out * width_out +
                     d_out * height_out * width_out +
                     h_out * width_out +
                     w_out)
    tl.store(output_ptr + output_offset, accumulator)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            out_channels // groups, 
            *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, depth_in, height_in, width_in = x.shape
        assert in_channels == self.in_channels
        
        depth_out = (depth_in - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        height_out = (height_in - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        # Fix width calculation (was using height_in instead of width_in)
        width_out = (width_in - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]
        
        out_channels_per_group = self.out_channels // self.groups
        in_channels_per_group = self.in_channels // self.groups
        
        output = torch.empty(
            batch_size, 
            self.out_channels, 
            depth_out, 
            height_out, 
            width_out, 
            device=x.device, 
            dtype=x.dtype
        )
        
        grid = (self.groups * batch_size * out_channels_per_group, depth_out, height_out * width_out)
        
        _conv_transpose3d_kernel[grid](
            x, output, self.weight,
            self.bias if self.bias is not None else x.new_empty(0),
            in_channels_per_group,
            out_channels_per_group,
            depth_in, height_in, width_in,
            depth_out, height_out, width_out,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.bias is not None,
            BLOCK_SIZE_CH=128
        )
        
        return output

# Test code
import math
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]
# =================== EVOLVE-BLOCK-END ===================