# ================== EVOLVE-BLOCK-START ==================
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
    # Meta-parameters
    eps: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)
    
    # Compute group boundaries
    group_start = pid_group * GROUP_SIZE
    group_range = tl.arange(0, GROUP_SIZE)
    feature_mask = group_range < (out_features - group_start)
    
    # Initialize accumulator
    acc = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
    x_ptrs = x_ptr + pid_batch * in_features
    
    # Loop over input features with vectorized loads
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        
        # Vectorized load
        x_block = tl.load(
            x_ptrs + k_offsets,
            mask=k_mask,
            other=0.0
        )
        
        # Weight pointers with precomputation
        w_start = (group_start + group_range[:, None]) * in_features
        w_ptrs = gemm_weight_ptr + w_start + k_offsets[None, :]
        w_block = tl.load(
            w_ptrs,
            mask=k_mask[None, :] & feature_mask[:, None],
            other=0.0
        )
        
        # Accumulate with efficient dot product
        acc += tl.sum(w_block * x_block[None, :], axis=1)
    
    # Add bias
    bias_ptrs = gemm_bias_ptr + group_start + group_range
    bias = tl.load(bias_ptrs, mask=feature_mask, other=0.0)
    acc += bias
    
    # Compute GroupNorm statistics (corrected group size)
    group_count = tl.sum(tl.cast(feature_mask, tl.float32))
    mean = tl.sum(acc) / group_count
    diff = acc - mean
    var = tl.sum(diff * diff) / group_count
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply GroupNorm with fused operations
    gn_weight = tl.load(gn_weight_ptr + group_start + group_range, mask=feature_mask, other=0.0)
    gn_bias = tl.load(gn_bias_ptr + group_start + group_range, mask=feature_mask, other=0.0)
    normalized = (diff * rstd) * gn_weight + gn_bias
    
    # Fused Swish operations
    sig1 = tl.sigmoid(normalized)
    swish1 = normalized * sig1
    
    # Multiply
    mult_weight = tl.load(mult_weight_ptr + group_start + group_range, mask=feature_mask, other=0.0)
    multiplied = swish1 * mult_weight
    
    # Second Swish (fused)
    sig2 = tl.sigmoid(multiplied)
    swish2 = multiplied * sig2
    
    # Store results with coalesced writes
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
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        output = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)
        
        # Launch kernel with optimized parameters
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
            eps=1e-5,
            BLOCK_SIZE_K=256,  # Increased for better memory efficiency
            GROUP_SIZE=group_size
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