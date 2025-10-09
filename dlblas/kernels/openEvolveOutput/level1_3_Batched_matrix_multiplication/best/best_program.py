# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_stride_A, M_stride_A, K_stride_A,
    batch_stride_B, K_stride_B, N_stride_B,
    batch_stride_C, M_stride_C, N_stride_C,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptr += pid_batch * batch_stride_A
    B_ptr += pid_batch * batch_stride_B
    C_ptr += pid_batch * batch_stride_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_valid = tl.minimum(BLOCK_K, k_remaining)
        
        a = tl.load(
            A_ptr + offs_m[:, None] * M_stride_A + (k * BLOCK_K + offs_k[None, :]) * K_stride_A,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_valid),
            other=0.0
        )
        b = tl.load(
            B_ptr + (k * BLOCK_K + offs_k[:, None]) * K_stride_B + offs_n[None, :] * N_stride_B,
            mask=(offs_k[:, None] < k_valid) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b, allow_tf32=False)  # Force full FP32 precision
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * M_stride_C + offs_cn[None, :] * N_stride_C
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        batch_size, M, K = A.shape
        _, K, N = B.shape
        
        C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
        
        grid = lambda META: (
            batch_size,
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N'])
        )
        
        bmm_kernel[grid](
            A, B, C,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            M, N, K
        )
        return C

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k, device='cuda')
    B = torch.randn(batch_size, k, n, device='cuda')
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================