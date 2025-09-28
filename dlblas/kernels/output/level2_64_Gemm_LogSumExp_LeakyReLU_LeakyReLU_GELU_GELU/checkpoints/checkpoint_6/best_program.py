# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _logsumexp_activations_kernel(
    x_ptr,
    output_ptr,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_stride
    
    # Initialize max and exp_sum for online logsumexp
    row_max = float('-inf')
    exp_sum = 0.0
    
    # Single-pass online logsumexp computation
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        ptr = x_ptr + row_start + col_idx
        vals = tl.load(ptr, mask=mask, other=float('-inf'))
        
        # Compute local max
        local_max = tl.max(vals, axis=0)
        new_max = tl.maximum(row_max, local_max)
        
        # Compute exp values and block sum
        exp_vals = tl.exp(vals - new_max)
        block_sum = tl.sum(exp_vals, axis=0)
        
        # Update global sum with scale factor
        exp_sum = exp_sum * tl.exp(row_max - new_max) + block_sum
        row_max = new_max
    
    # Compute final logsumexp
    log_sum_exp = row_max + tl.log(exp_sum)
    
    # LeakyReLU twice
    y = tl.maximum(0.01 * log_sum_exp, log_sum_exp)
    y = tl.maximum(0.01 * y, y)
    
    # GELU twice using erf approximation
    gelu_coeff = 0.5 * (1.0 + tl.erf(y * 0.7071067811865475))
    y = y * gelu_coeff
    y = y * 0.5 * (1.0 + tl.erf(y * 0.7071067811865475))
    
    # Store final result
    tl.store(output_ptr + row_idx, y)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        # GEMM operation
        x = self.linear(x)
        
        # Prepare for kernel launch
        x = x.contiguous()
        output = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)
        
        # Kernel configuration
        grid = (x.size(0),)
        n_cols = x.size(1)
        
        # Launch kernel with optimized block size
        _logsumexp_activations_kernel[grid](
            x,
            output,
            x.stride(0),
            n_cols,
            BLOCK_SIZE=triton.next_power_of_2(n_cols)
        )
        return output

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================