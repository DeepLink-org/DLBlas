# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_activations_kernel(
    x_ptr,
    add_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_id = tl.program_id(1)
    col_start = block_id * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    
    row_start = row_idx * n_cols
    x_ptrs = x_ptr + row_start + col_offsets
    add_ptrs = add_ptr + col_offsets
    
    # Vectorized loads with cache optimization
    x_vals = tl.load(x_ptrs, mask=col_mask, other=0.0, cache_modifier=".cg")
    add_vals = tl.load(add_ptrs, mask=col_mask, other=0.0, cache_modifier=".cg")
    
    # Fused operations
    y = x_vals + add_vals
    swish = y * tl.sigmoid(y)
    tanh = 2 * tl.sigmoid(2 * swish) - 1
    gelu = 0.5 * tanh * (1.0 + tl.erf(tanh * 0.7071067811865475))
    hardtanh = tl.minimum(tl.maximum(gelu, -1.0), 1.0)
    
    # Vectorized store with cache optimization
    tl.store(output_ptr + row_start + col_offsets, hardtanh, mask=col_mask, cache_modifier=".cg")

def fused_activations(x, add_value):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 128  # Optimized fixed block size for H100
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
    num_warps = 4  # 128 threads / 32 threads/warp = 4 warps
    
    _fused_activations_kernel[grid](
        x, add_value, output, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.matmul(x)
        x = fused_activations(x, self.add_value)
        return x

batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
# =================== EVOLVE-BLOCK-END ===================