# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_relu_group_norm(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    batch_size,
    out_channels,
    D, H, W,
    groups,
    eps: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    group_size = out_channels // groups
    num_elements = group_size * D * H * W
    
    # First pass: Compute mean
    sum_val = 0.0
    for idx in range(num_elements):
        c = idx // (D * H * W)
        spatial_idx = idx % (D * H * W)
        d = spatial_idx // (H * W)
        hw = spatial_idx % (H * W)
        h = hw // W
        w = hw % W
        
        channel_index = pid_group * group_size + c
        offset = pid_batch * out_channels * D * H * W + channel_index * D * H * W + d * H * W + h * W + w
        x = tl.load(input_ptr + offset)
        x_relu = tl.where(x > 0, x, 0.0)
        sum_val += x_relu
    mean = sum_val / num_elements
    
    # Second pass: Compute variance and normalize
    sum_sq = 0.0
    for idx in range(num_elements):
        c = idx // (D * H * W)
        spatial_idx = idx % (D * H * W)
        d = spatial_idx // (H * W)
        hw = spatial_idx % (H * W)
        h = hw // W
        w = hw % W
        
        channel_index = pid_group * group_size + c
        offset = pid_batch * out_channels * D * H * W + channel_index * D * H * W + d * H * W + h * W + w
        x = tl.load(input_ptr + offset)
        x_relu = tl.where(x > 0, x, 0.0)
        centered = x_relu - mean
        sum_sq += centered * centered
    
    variance = sum_sq / num_elements
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Third pass: Apply normalization
    for idx in range(num_elements):
        c = idx // (D * H * W)
        spatial_idx = idx % (D * H * W)
        d = spatial_idx // (H * W)
        hw = spatial_idx % (H * W)
        h = hw // W
        w = hw % W
        
        channel_index = pid_group * group_size + c
        offset = pid_batch * out_channels * D * H * W + channel_index * D * H * W + d * H * W + h * W + w
        x = tl.load(input_ptr + offset)
        x_relu = tl.where(x > 0, x, 0.0)
        normalized = (x_relu - mean) * inv_std
        gamma = tl.load(weight_ptr + channel_index)
        beta = tl.load(bias_ptr + channel_index)
        out_val = normalized * gamma + beta
        tl.store(output_ptr + offset, out_val)

def fused_relu_group_norm(x, weight, bias, groups, eps=1e-5):
    batch_size, out_channels, D, H, W = x.shape
    y = torch.empty_like(x)
    grid = (batch_size, groups)
    _fused_relu_group_norm[grid](
        x, y, weight, bias, 
        batch_size, out_channels, D, H, W, groups, eps
    )
    return y

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_relu_group_norm(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.groups
        )
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
# =================== EVOLVE-BLOCK-END ===================