# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_clamp_multiply_max(
    x_ptr,
    multiplier_ptr,
    output_ptr,
    clamp_min,
    clamp_max,
    out_channels,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    stride_mult_c,
    stride_b_out, stride_d_out, stride_h_out, stride_w_out,
    batch_size, depth, height, width, total_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_spatial

    w = offs % width
    h = (offs // width) % height
    d = (offs // (width * height)) % depth
    b = (offs // (width * height * depth)) % batch_size

    block_max = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for c in range(out_channels):
        off_x = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w
        val = tl.load(x_ptr + off_x, mask=mask, other=0)
        val = tl.minimum(tl.maximum(val, clamp_min), clamp_max)
        scale = tl.load(multiplier_ptr + c * stride_mult_c)
        val = val * scale
        block_max = tl.maximum(block_max, val)

    off_out = b * stride_b_out + d * stride_d_out + h * stride_h_out + w * stride_w_out
    tl.store(output_ptr + off_out, block_max, mask=mask)

def fused_clamp_multiply_max(x, multiplier, clamp_min, clamp_max):
    batch_size, out_channels, depth, height, width = x.shape
    output = torch.empty(batch_size, depth, height, width, device=x.device, dtype=x.dtype)
    
    if output.numel() == 0:
        return output

    multiplier_flat = multiplier.view(out_channels)
    total_spatial = batch_size * depth * height * width
    stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
    stride_b_out, stride_d_out, stride_h_out, stride_w_out = output.stride()
    stride_mult_c = multiplier_flat.stride(0)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_spatial, BLOCK_SIZE),)
    
    _fused_clamp_multiply_max[grid](
        x, multiplier_flat, output,
        clamp_min, clamp_max,
        out_channels,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        stride_mult_c,
        stride_b_out, stride_d_out, stride_h_out, stride_w_out,
        batch_size, depth, height, width, total_spatial,
        BLOCK_SIZE=BLOCK_SIZE
    )
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