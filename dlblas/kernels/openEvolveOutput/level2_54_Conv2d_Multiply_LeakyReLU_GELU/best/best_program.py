# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_ops_kernel(
    x_ptr,
    multiplier_ptr,
    output_ptr,
    negative_slope,
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_W: tl.constexpr,
):
    # Combine batch and channel dimensions
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w_block = tl.program_id(2)
    
    # Calculate batch and channel indices
    pid_b = pid_bc // C
    pid_c = pid_bc % C
    
    w_start = pid_w_block * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask = w_offsets < W
    
    base = pid_b * stride_b + pid_c * stride_c + pid_h * stride_h
    ptrs = base + w_offsets * stride_w
    
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    multiplier = tl.load(multiplier_ptr + pid_c)
    
    x = x * multiplier
    x = tl.where(x >= 0, x, x * negative_slope)
    x = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865475))
    
    tl.store(output_ptr + ptrs, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        output = torch.empty_like(x)
        neg_slope = self.leaky_relu.negative_slope
        
        stride_b, stride_c, stride_h, stride_w = x.stride()
        multiplier_flat = self.multiplier.view(-1).contiguous()
        
        BLOCK_W = 128
        # Use 3D grid: (B*C, H, blocks_w)
        grid = (B * C, H, (W + BLOCK_W - 1) // BLOCK_W)
        
        _fused_ops_kernel[grid](
            x, multiplier_flat, output, neg_slope,
            B, C, H, W,
            stride_b, stride_c, stride_h, stride_w,
            BLOCK_W=BLOCK_W
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
# =================== EVOLVE-BLOCK-END ===================