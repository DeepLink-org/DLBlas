# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    pool_size: tl.constexpr,
    scale_factor,
    stride_xm, stride_xn,
    stride_wm, stride_wn,
    stride_om,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(out_features // pool_size, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_b = pid // num_pid_m
    
    # Create pointers for the current batch
    x_ptr += pid_b * stride_xm
    output_ptr += pid_b * stride_om
    
    # Define linear rows for full features
    linear_rm = pid_m * (BLOCK_SIZE_M * pool_size) + tl.arange(0, BLOCK_SIZE_M * pool_size)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M * pool_size,), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_remaining = in_features - k
        k_mask = rk < k_remaining
        
        # Load input segment
        x = tl.load(x_ptr + k + rk, mask=k_mask, other=0.0)
        
        # Load weight tile
        w_ptrs = weight_ptr + linear_rm[:, None] * stride_wm + (k + rk[None, :]) * stride_wn
        w_mask = (linear_rm < out_features)[:, None] & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(w * x[None, :], axis=1)
    
    # Add bias
    bias = tl.load(bias_ptr + linear_rm, mask=linear_rm < out_features, other=0.0)
    acc += bias
    
    # Average pooling
    acc_2d = tl.reshape(acc, (BLOCK_SIZE_M, pool_size))
    pooled = tl.sum(acc_2d, axis=1) / pool_size
    
    # Apply GELU and scaling
    pooled = 0.5 * pooled * (1.0 + tl.erf(pooled * 0.7071067811865475))
    pooled = pooled * scale_factor
    
    # Store final result
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    tl.store(output_ptr + rm, pooled, mask=rm < (out_features // pool_size))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Validate divisible pooling
        if out_features % pool_kernel_size != 0:
            raise ValueError("out_features must be divisible by pool_kernel_size")
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, self.out_features // self.pool_kernel_size, 
                            device=x.device, dtype=x.dtype)
        
        # Configure kernel grid and launch
        grid = lambda meta: (batch_size * triton.cdiv(self.out_features // self.pool_kernel_size, meta['BLOCK_SIZE_M']),)
        
        fused_forward_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features, self.pool_kernel_size, self.scale_factor,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_K=128,
        )
        
        # Final max reduction
        return torch.max(output, dim=1).values

batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
# =================== EVOLVE-BLOCK-END ===================