# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_clamp_multiply_max_kernel(
    input_ptr,
    multiplier_ptr,
    output_ptr,
    clamp_min,
    clamp_max,
    C,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    stride_out_b,
    stride_out_d,
    stride_out_h,
    stride_out_w,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    channel_offsets = tl.arange(0, BLOCK_C)
    mask = channel_offsets < C
    
    base = pid_b * stride_b + pid_d * stride_d + pid_h * stride_h + pid_w * stride_w
    input_ptrs = input_ptr + base + channel_offsets * stride_c
    values = tl.load(input_ptrs, mask=mask, other=0.0)
    
    multiplier_vals = tl.load(multiplier_ptr + channel_offsets, mask=mask, other=0.0)
    
    clamped = tl.where(mask, 
                      tl.minimum(tl.maximum(values, clamp_min), clamp_max),
                      0.0)
    scaled = clamped * multiplier_vals
    scaled = tl.where(mask, scaled, -float('inf'))
    
    max_val = tl.max(scaled, axis=0)
    
    output_offset = pid_b * stride_out_b + pid_d * stride_out_d + pid_h * stride_out_h + pid_w * stride_out_w
    tl.store(output_ptr + output_offset, max_val)

def fused_clamp_multiply_max(x, multiplier, clamp_min, clamp_max):
    B, C, D, H, W = x.shape
    multiplier = multiplier.view(C)
    output = torch.empty((B, D, H, W), device=x.device, dtype=x.dtype)
    
    if x.is_cuda:
        stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
        stride_out_b, stride_out_d, stride_out_h, stride_out_w = output.stride()
        BLOCK_C = triton.next_power_of_2(C)
        
        grid = (B, D, H, W)
        _fused_clamp_multiply_max_kernel[grid](
            x, multiplier, output,
            clamp_min, clamp_max,
            C,
            stride_b, stride_c, stride_d, stride_h, stride_w,
            stride_out_b, stride_out_d, stride_out_h, stride_out_w,
            BLOCK_C
        )
    else:
        output = torch.clamp(x, clamp_min, clamp_max) * multiplier.view(1, C, 1, 1, 1)
        output = torch.max(output, dim=1)[0]
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = fused_clamp_multiply_max(x, self.multiplier, self.clamp_min, self.clamp_max)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
# =================== EVOLVE-BLOCK-END ===================