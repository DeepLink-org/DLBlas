# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
    ],
    key=['in_features', 'out_features'],
)
@triton.jit
def fused_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    in_features,
    out_features,
    subtract_value,
    multiply_value,
    stride_xb,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ob,
    stride_on,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_n = offs_n < out_features
    
    x_ptrs = x_ptr + pid_b * stride_xb + offs_k[:, None] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_remaining = in_features - k * BLOCK_SIZE_K
        k_valid = tl.minimum(BLOCK_SIZE_K, k_remaining)
        mask_k = offs_k < k_valid
        
        x = tl.load(x_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < out_features), other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.sum(w * x, axis=0)
        
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b
    
    acc = (acc - subtract_value) * multiply_value
    acc = tl.maximum(acc, 0.0)
    
    offs_on = pid_b * stride_ob + offs_n * stride_on
    tl.store(output_ptr + offs_on, acc, mask=mask_n)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty((batch_size, self.out_features), 
                             device=x.device, dtype=x.dtype)
        
        grid = (batch_size, triton.cdiv(self.out_features, 128))
        
        fused_linear_kernel[grid](
            x,
            self.weight,
            self.bias,
            output,
            self.in_features,
            self.out_features,
            self.subtract_value,
            self.multiply_value,
            x.stride(0),
            x.stride(1),
            self.weight.stride(0),
            self.weight.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,
        )
        return output

batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
# =================== EVOLVE-BLOCK-END ===================