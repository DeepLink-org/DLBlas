# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    x_ptr,
    output_ptr,
    n_features,
    spatial_size,
    eps,
    BLOCK_SPATIAL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_spatial_block = tl.program_id(1)
    
    spatial_offsets = pid_spatial_block * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    mask = spatial_offsets < spatial_size
    
    sum_sq = tl.zeros((BLOCK_SPATIAL,), dtype=tl.float32)
    
    for feature_idx in range(n_features):
        ptr = x_ptr + pid_batch * n_features * spatial_size + feature_idx * spatial_size + spatial_offsets
        x_val = tl.load(ptr, mask=mask, other=0.0)
        sum_sq += x_val * x_val
    
    rms = tl.sqrt(sum_sq / n_features + eps)
    
    for feature_idx in range(n_features):
        ptr = x_ptr + pid_batch * n_features * spatial_size + feature_idx * spatial_size + spatial_offsets
        out_ptr = output_ptr + pid_batch * n_features * spatial_size + feature_idx * spatial_size + spatial_offsets
        x_val = tl.load(ptr, mask=mask, other=0.0)
        normalized = x_val / rms
        tl.store(out_ptr, normalized, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        output = torch.empty_like(x)
        
        batch_size = x.shape[0]
        spatial_size = x.shape[2] * x.shape[3]
        
        BLOCK_SPATIAL = 128
        grid = (batch_size, triton.cdiv(spatial_size, BLOCK_SPATIAL))
        
        _rms_norm_kernel[grid](
            x, output,
            self.num_features,
            spatial_size,
            self.eps,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
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