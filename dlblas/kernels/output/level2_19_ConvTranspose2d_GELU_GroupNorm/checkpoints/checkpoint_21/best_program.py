# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    beta_ptr,
    n_channels,
    n_groups,
    eps,
    H,
    W,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    group_idx = pid % n_groups
    batch_idx = pid // n_groups
    channels_per_group = n_channels // n_groups

    # Compute mean and variance
    mean = 0.0
    var = 0.0
    count = 0.0

    for c in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
        for h in range(0, H, BLOCK_SIZE):
            for w in range(0, W, BLOCK_SIZE):
                h_offsets = h + tl.arange(0, BLOCK_SIZE)
                w_offsets = w + tl.arange(0, BLOCK_SIZE)
                
                # Create mask for valid elements
                h_mask = h_offsets < H
                w_mask = w_offsets < W
                mask = h_mask[:, None] & w_mask[None, :]
                
                # Calculate offsets
                offsets = batch_idx * stride_b + c * stride_c + h_offsets[:, None] * stride_h + w_offsets[None, :] * stride_w
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                
                # Update mean and variance
                block_sum = tl.sum(x)
                block_sq_sum = tl.sum(x * x)
                block_count = tl.sum(tl.where(mask, 1.0, 0.0))
                
                mean += block_sum
                var += block_sq_sum
                count += block_count

    mean = mean / count
    var = var / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    for c in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        
        for h in range(0, H, BLOCK_SIZE):
            for w in range(0, W, BLOCK_SIZE):
                h_offsets = h + tl.arange(0, BLOCK_SIZE)
                w_offsets = w + tl.arange(0, BLOCK_SIZE)
                
                # Create mask for valid elements
                h_mask = h_offsets < H
                w_mask = w_offsets < W
                mask = h_mask[:, None] & w_mask[None, :]
                
                # Calculate offsets
                offsets = batch_idx * stride_b + c * stride_c + h_offsets[:, None] * stride_h + w_offsets[None, :] * stride_w
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                
                # Normalize
                y = (x - mean) * inv_std
                y = y * gamma + beta
                tl.store(y_ptr + offsets, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.nn.functional.gelu(x)
        
        # Convert to channels-last for efficient memory access
        x = x.contiguous(memory_format=torch.channels_last)
        n, c, h, w = x.shape
        
        # Prepare output tensor
        y = torch.empty_like(x)
        
        # Launch Triton kernel for GroupNorm
        grid = (n * self.group_norm.num_groups,)
        group_norm_kernel[grid](
            x, y,
            self.group_norm.weight,
            self.group_norm.bias,
            c, self.group_norm.num_groups, self.group_norm.eps,
            h, w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            BLOCK_SIZE=16
        )
        return y

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 32, 32
kernel_size = 4
stride = 2
groups = 8
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
# =================== EVOLVE-BLOCK-END ===================