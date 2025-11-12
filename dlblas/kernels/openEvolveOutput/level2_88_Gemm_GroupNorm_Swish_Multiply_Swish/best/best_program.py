# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_forward_kernel(
    # Pointers
    x_ptr,
    gemm_weight_ptr,
    gemm_bias_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    mult_weight_ptr,
    output_ptr,
    # Shapes
    batch_size,
    in_features,
    out_features,
    num_groups,
    group_size,
    # Meta-parameters
    eps: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    # Compute group boundaries
    group_start = pid_group * group_size
    group_range = tl.arange(0, BLOCK_SIZE_N)
    feature_mask = group_range < group_size
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    x_ptr += pid_batch * in_features
    
    # Loop over input features with vectorized loads
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        
        # Vectorized load
        x_block = tl.load(
            x_ptr + k_offsets,
            mask=k_mask,
            other=0.0,
        )
        
        # Weight pointers
        w_ptrs = gemm_weight_ptr + (group_start + group_range[:, None]) * in_features + k_offsets[None, :]
        w_block = tl.load(
            w_ptrs,
            mask=k_mask[None, :] & feature_mask[:, None],
            other=0.0,
        )
        
        # Accumulate dot product
        acc += tl.sum(w_block * x_block[None, :], axis=1)
    
    # Add bias
    bias_ptrs = gemm_bias_ptr + group_start + group_range
    bias = tl.load(bias_ptrs, mask=feature_mask, other=0.0)
    acc += bias
    
    # Compute GroupNorm statistics
    group_count = tl.sum(tl.cast(feature_mask, tl.float32))
    mean = tl.sum(acc) / group_count
    diff = acc - mean
    var = tl.sum(diff * diff) / group_count
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply GroupNorm
    gn_weight = tl.load(gn_weight_ptr + group_start + group_range, mask=feature_mask, other=0.0)
    gn_bias = tl.load(gn_bias_ptr + group_start + group_range, mask=feature_mask, other=0.0)
    normalized = (diff * rstd) * gn_weight + gn_bias
    
    # Fused Swish operations
    sig1 = tl.sigmoid(normalized)
    swish1 = normalized * sig1
    
    # Multiply
    mult_weight = tl.load(mult_weight_ptr + group_start + group_range, mask=feature_mask, other=0.0)
    multiplied = swish1 * mult_weight
    
    # Second Swish
    sig2 = tl.sigmoid(multiplied)
    swish2 = multiplied * sig2
    
    # Store results
    out_offsets = pid_batch * out_features + group_start + group_range
    tl.store(output_ptr + out_offsets, swish2, mask=feature_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.out_features = out_features
        self.num_groups = num_groups

    def forward(self, x):
        batch_size, _ = x.shape
        group_size = self.out_features // self.num_groups
        P2 = int(2 ** math.ceil(math.log2(group_size)))
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        output = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)
        
        # Compute required number of warps
        min_warps = max(1, (P2 + 31) // 32)
        num_warps = 1
        while num_warps < min_warps:
            num_warps *= 2
            
        # Launch kernel
        grid = (batch_size, self.num_groups)
        _fused_forward_kernel[grid](
            x, 
            self.gemm.weight, 
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            output,
            batch_size,
            self.gemm.in_features,
            self.out_features,
            self.num_groups,
            group_size,
            eps=1e-5,
            BLOCK_SIZE_K=128,
            BLOCK_SIZE_N=P2,
            num_warps=num_warps,
            num_stages=2
        )
        return output

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
# =================== EVOLVE-BLOCK-END ===================