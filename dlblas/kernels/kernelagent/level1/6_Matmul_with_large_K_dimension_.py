import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Existing good configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 128}, num_stages=5, num_warps=4),
        # Added higher-K tiles and deeper pipelines for large-K performance
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Alignment/coalescing hints for better codegen on Hopper
    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_n, 8)
    tl.multiple_of(offs_k, 8)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # masks/constants for TF32-style rounding (round-to-nearest on 10-bit mantissa)
    SIGN_MASK = 1 << 31
    ABS_MASK = (1 << 31) - 1
    RND_ADD = 1 << 12
    TRUNC_MASK = ~((1 << 13) - 1)

    # Static masks for M/N bounds
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # Software-prefetch pipeline: load first K-slice
    k_iter = 0
    k_mask = (k_iter + offs_k) < K
    a_mask = mask_m & k_mask[None, :]
    b_mask = k_mask[:, None] & mask_n

    a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=".cg", eviction_policy="evict_first")
    b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".cg", eviction_policy="evict_last")

    while True:
        # Emulate TF32 inputs: keep sign/exponent; round mantissa to 10 bits (round-to-nearest)
        ai = tl.bitcast(a, tl.int32)
        bi = tl.bitcast(b, tl.int32)

        a_sign = ai & SIGN_MASK
        b_sign = bi & SIGN_MASK
        a_abs = ai & ABS_MASK
        b_abs = bi & ABS_MASK

        a_abs = a_abs + RND_ADD
        b_abs = b_abs + RND_ADD

        a_abs = a_abs & TRUNC_MASK
        b_abs = b_abs & TRUNC_MASK

        ai = a_abs | a_sign
        bi = b_abs | b_sign

        a_tf32 = tl.bitcast(ai, tl.float32)
        b_tf32 = tl.bitcast(bi, tl.float32)

        acc += tl.dot(a_tf32, b_tf32)

        # Advance K
        k_iter += BLOCK_K
        if k_iter >= K:
            break

        # Prefetch next tiles while computing
        a_ptrs_next = a_ptrs + BLOCK_K * stride_ak
        b_ptrs_next = b_ptrs + BLOCK_K * stride_bk

        k_mask = (k_iter + offs_k) < K
        a_mask = mask_m & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n

        a = tl.load(a_ptrs_next, mask=a_mask, other=0.0, cache_modifier=".cg", eviction_policy="evict_first")
        b = tl.load(b_ptrs_next, mask=b_mask, other=0.0, cache_modifier=".cg", eviction_policy="evict_last")

        a_ptrs = a_ptrs_next
        b_ptrs = b_ptrs_next

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        # Use Triton kernel when possible; otherwise, fall back to torch.matmul
        if (
            A.is_cuda and B.is_cuda and
            A.ndim == 2 and B.ndim == 2 and
            A.shape[1] == B.shape[0] and
            A.dtype == torch.float32 and B.dtype == torch.float32 and
            torch.backends.cuda.matmul.allow_tf32  # align with PyTorch default compute path on Hopper
        ):
            M, K = A.shape
            Kb, N = B.shape
            assert K == Kb, "Inner dimensions must match for matmul"

            C = torch.empty((M, N), device=A.device, dtype=A.dtype)

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

            _matmul_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
            )
            return C
        else:
            # Fallback ensures exact semantic parity with torch.matmul in unsupported cases
            return torch.matmul(A, B)

M = 256
N = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed