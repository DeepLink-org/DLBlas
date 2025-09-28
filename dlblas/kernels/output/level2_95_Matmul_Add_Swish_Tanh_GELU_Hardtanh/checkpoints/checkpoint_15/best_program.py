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
    n_rows,
    n_cols,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_start = pid_row * BLOCK_SIZE_R
    col_start = pid_col * BLOCK_SIZE_C
    
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_C)
    col_mask = col_offsets < n_cols
    
    # Preload add values for column block
    add_vals = tl.load(add_ptr + col_offsets, mask=col_mask, other=0.0)
    
    for r in range(BLOCK_SIZE_R):
        row_idx = row_start + r
        if row_idx < n_rows:
            # Load input values with vectorization
            x_ptrs = x_ptr + row_idx * n_cols + col_offsets
            x_vals = tl.load(x_ptrs, mask=col_mask, other=0.0)
            
            # Fused operations with optimized computation order
            y = x_vals + add_vals
            swish = y * tl.sigmoid(y)
            tanh = 2 * tl.sigmoid(2 * swish) - 1  # tanh approximation
            gelu = 0.5 * tanh * (1.0 + tl.erf(tanh * 0.7071067811865475))
            hardtanh = tl.minimum(tl.maximum(gelu, -1.0), 1.0)
            
            # Store results with vectorization
            out_ptrs = output_ptr + row_idx * n_cols + col_offsets
            tl.store(out_ptrs, hardtanh, mask=col_mask)

def fused_activations(x, add_value):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    
    # Optimized block sizes for H100 architecture
    BLOCK_SIZE_R = 4
    BLOCK_SIZE_C = 128
    
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_R), triton.cdiv(n_cols, BLOCK_SIZE_C))
    num_warps = 4  # Fixed warps for better occupancy
    
    _fused_activations_kernel[grid](
        x, add_value, output, n_rows, n_cols,
        BLOCK_SIZE_R=BLOCK_SIZE_R, 
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_warps=num_warps
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