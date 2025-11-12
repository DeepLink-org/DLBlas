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
    N, C, H, W,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    group_size = C // num_groups
    
    # Compute mean and variance per group
    mean = 0.0
    var = 0.0
    count = tl.zeros((), tl.uint32)  # Fixed type initialization
    
    # Loop through channels in group
    for c in range(g * group_size, (g + 1) * group_size):
        # Loop through spatial dimensions
        for h in range(0, H, BLOCK_SIZE):
            for w in range(0, W, BLOCK_SIZE):
                offsets = tl.arange(0, BLOCK_SIZE)
                h_offsets = h + offsets
                w_offsets = w + offsets
                
                mask = (h_offsets < H) & (w_offsets < W)
                x_ptrs = x_ptr + n * C * H * W + c * H * W + h_offsets[:, None] * W + w_offsets[None, :]
                x_vals = tl.load(x_ptrs, mask=mask[:, None] & mask[None, :])
                
                mean += tl.sum(x_vals)
                count += tl.sum(mask) * tl.sum(mask)
    
    mean = mean / count
    mean = mean.to(tl.float32)
    
    # Compute variance
    for c in range(g * group_size, (g + 1) * group_size):
        for h in range(0, H, BLOCK_SIZE):
            for w in range(0, W, BLOCK_SIZE):
                offsets = tl.arange(0, BLOCK_SIZE)
                h_offsets = h + offsets
                w_offsets = w + offsets
                
                mask = (h_offsets < H) & (w_offsets < W)
                x_ptrs = x_ptr + n * C * H * W + c * H * W + h_offsets[:, None] * W + w_offsets[None, :]
                x_vals = tl.load(x_ptrs, mask=mask[:, None] & mask[None, :])
                
                diff = x_vals - mean
                var += tl.sum(diff * diff)
    
    var = var / count
    std = tl.sqrt(var + eps)
    
    # Apply normalization
    for c in range(g * group_size, (g + 1) * group_size):
        for h in range(0, H, BLOCK_SIZE):
            for w in range(0, W, BLOCK_SIZE):
                offsets = tl.arange(0, BLOCK_SIZE)
                h_offsets = h + offsets
                w_offsets = w + offsets
                
                mask = (h_offsets < H) & (w_offsets < W)
                x_ptrs = x_ptr + n * C * H * W + c * H * W + h_offsets[:, None] * W + w_offsets[None, :]
                y_ptrs = y_ptr + n * C * H * W + c * H * W + h_offsets[:, None] * W + w_offsets[None, :]
                
                x_vals = tl.load(x_ptrs, mask=mask[:, None] & mask[None, :])
                normalized = (x_vals - mean) / std
                gamma_val = tl.load(gamma_ptr + c)
                beta_val = tl.load(beta_ptr + c)
                out = normalized * gamma_val + beta_val
                tl.store(y_ptrs, out, mask=mask[:, None] & mask[None, :])

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.num_groups = num_groups
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        
        # Custom group norm with Triton
        N, C, H, W = x.shape
        y = torch.empty_like(x)
        eps = 1e-5
        BLOCK_SIZE = 16
        
        grid = (N, self.num_groups)
        group_norm_kernel[grid](
            x, y, self.gamma, self.beta,
            N, C, H, W,
            self.num_groups, eps, BLOCK_SIZE
        )
        return y

batch_size = 128
in_channels = 32
out_channels = 64
kernel_size = 4
stride = 2
padding = 1
groups = 8
num_groups = 4
height, width = 32, 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
# =================== EVOLVE-BLOCK-END ===================