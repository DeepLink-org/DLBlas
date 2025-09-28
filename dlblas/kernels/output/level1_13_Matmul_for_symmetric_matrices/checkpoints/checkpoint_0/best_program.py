# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
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
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        
        A = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            k_effective = min(BLOCK_K, k_remaining)
            
            a = tl.load(A, mask=rk[None, :] < k_effective, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k_effective, other=0.0)
            
            acc += tl.dot(a, b, out_dtype=tl.float32)
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk
        
        rm_mask = rm < M
        rn_mask = rn < N
        mask = rm_mask[:, None] & rn_mask[None, :]
        
        c = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        tl.store(c, acc, mask=mask)

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        
        self.matmul_kernel[grid](
            A, B, C,
            M, N, K,
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