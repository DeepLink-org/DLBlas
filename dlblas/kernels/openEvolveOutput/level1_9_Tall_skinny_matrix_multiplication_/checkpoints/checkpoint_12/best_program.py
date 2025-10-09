# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, K)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    mask_a = (offs_m[:, None] < M) & (tl.arange(0, K)[None, :] < K)
    mask_b = (tl.arange(0, K)[:, None] < K) & (offs_n[None, :] < N)
    
    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)
    
    c = tl.dot(a, b)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        M, K = A.shape
        K_B, N = B.shape
        assert K == K_B, f"Inner dimension mismatch: A has {K}, B has {K_B}"
        
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        if M * N * K == 0:
            return C
        
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']), 
            triton.cdiv(N, META['BLOCK_N']),
        )
        
        matmul_kernel[grid](
            A, B, C,
            M, K, N,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
        
        return C

M = 16384
N = 16

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================