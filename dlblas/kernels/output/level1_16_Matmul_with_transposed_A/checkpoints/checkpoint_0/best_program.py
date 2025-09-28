# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    A = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (rm[:, None] < M) & (rk[None, :] < k_remaining)
        b_mask = (rk[:, None] < k_remaining) & (rn[None, :] < N)
        
        a = tl.load(A, mask=a_mask, other=0.0)
        b = tl.load(B, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=True)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    
    c = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c, acc, mask=c_mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape[1], A.shape[0]
        N = B.shape[1]
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(1), A.stride(0),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        return C

def get_inputs():
    A = torch.randn(4096, 1024, device='cuda')
    B = torch.randn(4096, 2048, device='cuda')
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================