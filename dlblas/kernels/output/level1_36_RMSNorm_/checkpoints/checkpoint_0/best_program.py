# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _rms_norm_forward(
    x_ptr,
    output_ptr,
    n_features,
    n_spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    offsets = pid_b * n_spatial * n_features + pid_s + tl.arange(0, BLOCK_SIZE) * n_spatial
    mask = tl.arange(0, BLOCK_SIZE) < n_features
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = tl.where(mask, x * x, 0.0)
    
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / n_features
    rms = tl.sqrt(mean_sq + eps)
    
    output = x / rms
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        output = torch.empty_like(x)
        batch_size, _, dim1, dim2 = x.shape
        n_spatial = dim1 * dim2
        
        grid = (batch_size, n_spatial)
        BLOCK_SIZE = min(triton.next_power_of_2(self.num_features), 128)
        
        _rms_norm_forward[grid](
            x, output, 
            self.num_features, 
            n_spatial,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        return output

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
# =================== EVOLVE-BLOCK-END ===================