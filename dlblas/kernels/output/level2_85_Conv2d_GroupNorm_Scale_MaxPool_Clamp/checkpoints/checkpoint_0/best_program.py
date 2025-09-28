# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gn_scale_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    s_ptr,
    y_ptr,
    C, G, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    
    group_size = C // G
    start = g * group_size
    offsets = tl.arange(0, BLOCK_SIZE)
    c_idx = offsets // (H * W)
    spatial_idx = offsets % (H * W)
    ch_offset = start + c_idx
    
    mask = offsets < group_size * H * W
    base_ptr = n * C * H * W + start * H * W
    
    # Load data
    x = tl.load(x_ptr + base_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / (group_size * H * W)
    
    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / (group_size * H * W)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and scale
    w = tl.load(w_ptr + ch_offset, mask=c_idx < group_size, other=0.0)
    b = tl.load(b_ptr + ch_offset, mask=c_idx < group_size, other=0.0)
    scale = tl.load(s_ptr + ch_offset, mask=c_idx < group_size, other=0.0)
    
    y = (x_centered * rstd) * w + b
    y = y * scale
    tl.store(y_ptr + base_ptr + offsets, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        
        # Fused group norm and scaling with Triton
        N, C, H, W = x.shape
        G = self.group_norm.num_groups
        y = torch.empty_like(x)
        group_elements = (C // G) * H * W
        grid = (N, G)
        fused_gn_scale_kernel[grid](
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.scale.view(-1), 
            y,
            C, G, H, W,
            self.group_norm.eps,
            BLOCK_SIZE=triton.next_power_of_2(group_elements)
        )
        
        x = self.maxpool(y)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
# =================== EVOLVE-BLOCK-END ===================