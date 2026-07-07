# Kernel 1 & 2: q_projection and weights_projection (MatMul kernels)
# Written in Triton-Ascend, targeting Ascend910B2 Cube unit
#
# These kernels implement tiled matrix multiplication:
#   C(M,N) = A(M,K) @ B(K,N)
# with bf16 inputs/outputs and fp32 accumulation.

import torch
import triton
import triton.language as tl


# ==============================================================================
# Generic Tiled MatMul Kernel (used by both q_projection and weights_projection)
# ==============================================================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matrix multiplication: C = A @ B
    A: (M, K) row-major
    B: (K, N) row-major
    C: (M, N) row-major
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask_a = (offs_k[None, :] < k_remaining) & (offs_m[:, None] < M)
        k_mask_b = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# ==============================================================================
# Kernel 1: q_projection — qr @ wq_weight^T
# ==============================================================================

def q_projection(qr: torch.Tensor, wq_weight: torch.Tensor) -> torch.Tensor:
    """Kernel 1: Q linear projection.

    Computes: q_flat = qr @ wq_weight^T

    Args:
        qr: (B*S, q_lora_rank) bf16, row-major
        wq_weight: (n_heads*head_dim, q_lora_rank) bf16, row-major (will be transposed)

    Returns:
        q_flat: (B*S, n_heads*head_dim) bf16
    """
    if qr.dim() == 3:
        B, S, K = qr.shape
        qr = qr.reshape(-1, K)
    M, K = qr.shape  # M = B*S, K = q_lora_rank
    N, K2 = wq_weight.shape  # N = H*D
    assert K == K2, f"Shape mismatch: qr K={K}, wq K={K2}"

    q_flat = torch.empty(M, N, dtype=qr.dtype, device=qr.device)

    # wq_weight is (N, K), we need (K, N) for matmul
    # Transpose wq_weight before passing to kernel
    wq_t = wq_weight.t().contiguous()  # (K, N)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        qr, wq_t, q_flat,
        M, N, K,
        qr.stride(0), qr.stride(1),
        wq_t.stride(0), wq_t.stride(1),
        q_flat.stride(0), q_flat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return q_flat


# ==============================================================================
# Kernel 2: weights_projection — x @ weights_proj_weight^T
# ==============================================================================

def weights_projection(x: torch.Tensor, w_weight: torch.Tensor) -> torch.Tensor:
    """Kernel 2: Weights linear projection.

    Computes: weights = x @ w_weight^T

    Args:
        x: (B*S, dim) bf16, row-major
        w_weight: (n_heads, dim) bf16, row-major

    Returns:
        weights: (B*S, n_heads) bf16
    """
    if x.dim() == 3:
        B, S, K = x.shape
        x = x.reshape(-1, K)
    M, K = x.shape  # M = B*S, K = dim
    N, K2 = w_weight.shape  # N = n_heads
    assert K == K2, f"Shape mismatch: x K={K}, w K={K2}"

    weights = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Transpose weight: (N, K) → (K, N)
    w_t = w_weight.t().contiguous()  # (K, N)

    BLOCK_M = 64
    BLOCK_N = 16
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        x, w_t, weights,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        weights.stride(0), weights.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return weights
