# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=2, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_forward_kernel(
    x_ptr, w_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < K
        
        x_vals = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=k_mask[None, :] & (offs_m[:, None] < M),
            other=0.0
        )
        w_vals = tl.load(
            w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(x_vals, w_vals)
    
    b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    
    swish = acc * tl.sigmoid(acc)
    div = swish / 2.0
    clamp1 = tl.minimum(tl.maximum(div, -1.0), 1.0)
    
    # Replace unsupported tl.tanh with equivalent exp/log implementation
    exp_2x = tl.exp(2 * clamp1)
    tanh = (exp_2x - 1) / (exp_2x + 1)
    
    clamp2 = tl.minimum(tl.maximum(tanh, -1.0), 1.0)
    
    offs_out = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        output_ptr + offs_out, 
        clamp2,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):
        M, K = x.shape
        N = self.gemm.out_features
        
        x = x.contiguous()
        weight_t = self.gemm.weight.t().contiguous()
        bias = self.gemm.bias if self.gemm.bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']), 
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        _fused_forward_kernel[grid](
            x, weight_t, bias, output,
            M, N, K,
            x.stride(0), x.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            output.stride(0), output.stride(1),
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