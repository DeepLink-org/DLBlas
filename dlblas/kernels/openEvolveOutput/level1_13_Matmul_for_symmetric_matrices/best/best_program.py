# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    N, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    
    offs_m = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < N - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)  # Disabled TF32 for FP32 precision
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < N) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        N = A.size(0)
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)
        
        grid = lambda META: (
            triton.cdiv(N, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        matmul_kernel[grid](
            A, B, C, 
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        return C

N = 4096

def get_inputs():
    A = torch.randn(N, N)
    A = (A + A.T) / 2
    B = torch.randn(N, N)
    B = (B + B.T) / 2
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================