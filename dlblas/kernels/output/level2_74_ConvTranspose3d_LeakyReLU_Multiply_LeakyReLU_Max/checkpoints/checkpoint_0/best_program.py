# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_leaky_mul_leaky_kernel(
    x_ptr,
    multiplier_ptr,
    output_ptr,
    N, C, D, H, W,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    
    off_w = tl.arange(0, BLOCK_W)
    w_offsets = off_w
    mask = w_offsets < W
    
    base = pid_n * C * D * H * W + pid_c * D * H * W + pid_d * H * W + pid_h * W + w_offsets
    
    multiplier_val = tl.load(multiplier_ptr + pid_c)
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    
    leaky1 = tl.where(x >= 0, x, 0.2 * x)
    temp = leaky1 * multiplier_val
    out = tl.where(temp >= 0, temp, 0.2 * temp)
    
    tl.store(output_ptr + base, out, mask=mask)

def fused_leaky_mul_leaky(x, multiplier):
    N, C, D, H, W = x.shape
    multiplier = multiplier.contiguous().view(-1)
    output = torch.empty_like(x)
    grid = (N, C, D, H)
    BLOCK_W = min(128, W)
    _fused_leaky_mul_leaky_kernel[grid](x, multiplier, output, N, C, D, H, W, BLOCK_W=BLOCK_W)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        if not x.is_contiguous():
            x = x.contiguous()
        x = fused_leaky_mul_leaky(x, self.multiplier)
        x = self.max_pool(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
# =================== EVOLVE-BLOCK-END ===================