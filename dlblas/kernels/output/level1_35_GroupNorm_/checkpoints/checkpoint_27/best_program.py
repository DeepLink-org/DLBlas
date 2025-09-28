# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_forward_kernel(
    x_ptr,
    y_ptr,
    gamma_ptr,
    beta_ptr,
    n_elements_per_group,
    n_groups,
    n_channels,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group_idx = pid % n_groups
    sample_idx = pid // n_groups

    channels_per_group = n_channels // n_groups
    spatial_size = H * W
    group_size = channels_per_group * spatial_size

    base_offset = sample_idx * n_channels * spatial_size + group_idx * channels_per_group * spatial_size

    # First pass: Compute mean
    sum_val = 0.0
    for offset in range(0, group_size, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_size
        data = tl.load(x_ptr + base_offset + offs, mask=mask, other=0.0)
        sum_val += tl.sum(data, axis=0)
    mean = sum_val / group_size

    # Second pass: Compute variance
    var_val = 0.0
    for offset in range(0, group_size, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_size
        data = tl.load(x_ptr + base_offset + offs, mask=mask, other=0.0)
        diff = data - mean
        var_val += tl.sum(diff * diff, axis=0)
    var = var_val / group_size
    rstd = 1.0 / tl.sqrt(var + eps)

    # Third pass: Normalize and apply affine transformation
    for offset in range(0, group_size, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_size
        data = tl.load(x_ptr + base_offset + offs, mask=mask, other=0.0)
        
        # Normalize
        normalized = (data - mean) * rstd
        
        # Compute channel index
        spatial_idx = offs % spatial_size
        channel_in_group = offs // spatial_size
        global_channel_idx = group_idx * channels_per_group + channel_in_group
        
        # Load gamma and beta
        gamma = tl.load(gamma_ptr + global_channel_idx)
        beta = tl.load(beta_ptr + global_channel_idx)
        
        # Apply affine transformation
        out = normalized * gamma + beta
        
        tl.store(y_ptr + base_offset + offs, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        y = torch.empty_like(x)
        
        grid = (n * self.num_groups,)
        
        group_norm_forward_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            gamma_ptr=self.weight,
            beta_ptr=self.bias,
            n_elements_per_group=c // self.num_groups * h * w,
            n_groups=self.num_groups,
            n_channels=c,
            H=h,
            W=w,
            eps=self.eps,
            BLOCK_SIZE=1024
        )
        
        return y

batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups]
# =================== EVOLVE-BLOCK-END ===================