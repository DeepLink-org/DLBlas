# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_div_kernel(
    x_ptr,
    n_elements_per_channel,
    eps,
    divide_by,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * n_elements_per_channel
    
    # Step 1: Compute mean
    mean = 0.0
    for off in range(0, n_elements_per_channel, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements_per_channel
        data = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        mean += tl.sum(data, axis=0)
    mean = mean / n_elements_per_channel

    # Step 2: Compute variance
    var = 0.0
    for off in range(0, n_elements_per_channel, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements_per_channel
        data = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        centered = data - mean
        var += tl.sum(centered * centered, axis=0)
    var = var / n_elements_per_channel
    
    # Optimized normalization using rsqrt
    rstd = tl.rsqrt(var + eps)
    scale = rstd / divide_by

    # Step 3: Apply normalization and scaling
    for off in range(0, n_elements_per_channel, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements_per_channel
        data = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        centered = data - mean
        out = centered * scale
        tl.store(x_ptr + base + idx, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        batch, channels, height, width = x.shape
        n_elements_per_channel = height * width
        grid = (batch * channels,)
        instance_norm_div_kernel[grid](
            x, 
            n_elements_per_channel,
            1e-5,
            self.divide_by,
            BLOCK_SIZE=1024
        )
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
# =================== EVOLVE-BLOCK-END ===================