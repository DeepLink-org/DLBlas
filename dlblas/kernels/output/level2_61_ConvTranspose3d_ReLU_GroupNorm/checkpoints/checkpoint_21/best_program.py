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
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    group_start = pid_group * GROUP_SIZE
    spatial_size = D * H * W
    group_elements = GROUP_SIZE * spatial_size
    base_offset = pid_batch * out_channels * spatial_size + group_start * spatial_size
    
    # First pass: compute sum and sum_sq
    sum_val = 0.0
    sum_sq = 0.0
    for idx in range(0, group_elements, BLOCK_SIZE):
        offsets = base_offset + idx + tl.arange(0, BLOCK_SIZE)
        mask = (idx + tl.arange(0, BLOCK_SIZE)) < group_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        x_relu = tl.where(x > 0, x, 0.0)
        sum_val += tl.sum(x_relu, axis=0)
        sum_sq += tl.sum(x_relu * x_relu, axis=0)
    
    mean = sum_val / group_elements
    variance = tl.maximum(sum_sq / group_elements - mean * mean, 0.0)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Second pass: normalize and store
    for idx in range(0, group_elements, BLOCK_SIZE):
        offsets = base_offset + idx + tl.arange(0, BLOCK_SIZE)
        mask = (idx + tl.arange(0, BLOCK_SIZE)) < group_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        x_relu = tl.where(x > 0, x, 0.0)
        normalized = (x_relu - mean) * inv_std
        
        # Compute channel indices and load weights
        c_idx = (idx + tl.arange(0, BLOCK_SIZE)) // spatial_size
        c_global = group_start + c_idx
        w = tl.load(weight_ptr + c_global, mask=mask)
        b = tl.load(bias_ptr + c_global, mask=mask)
        
        out_val = normalized * w + b
        tl.store(output_ptr + offsets, out_val, mask=mask)

def fused_relu_group_norm(x, weight, bias, groups, eps=1e-5):
    x = x.contiguous()
    batch_size, out_channels, D, H, W = x.shape
    group_size = out_channels // groups
    y = torch.empty_like(x)
    grid = (batch_size, groups)
    _fused_relu_group_norm[grid](
        x, y, weight, bias, 
        batch_size, out_channels, D, H, W, groups, eps,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=1024
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