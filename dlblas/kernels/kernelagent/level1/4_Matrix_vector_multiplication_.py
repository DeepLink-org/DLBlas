import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemv_rowblock_kernel(
    A_ptr,  # *A: (M, K)
    B_ptr,  # *B: (K, 1)
    C_ptr,  # *C: (M, 1)
    M: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program id along output rows
    pid_m = tl.program_id(axis=0)
    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_m = rows < M

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Prepare base row offsets and K-range for tiling
    row_offs = rows[:, None] * stride_am
    k_range = tl.arange(0, BLOCK_K)

    # Double-buffered K-loop to improve overlap of loads and compute
    k0 = 0
    # Prefetch first tile
    ks0 = k0 + k_range
    mask_k0 = ks0 < K
    a_ptrs0 = A_ptr + row_offs + ks0[None, :] * stride_ak
    b_ptrs0 = B_ptr + ks0 * stride_bk + 0 * stride_bn
    a_tile0 = tl.load(a_ptrs0, mask=mask_m[:, None] & mask_k0[None, :], other=0.0)
    b_tile0 = tl.load(b_ptrs0, mask=mask_k0, other=0.0)
    k0 += BLOCK_K

    while k0 < K:
        # Prefetch next tile
        ks1 = k0 + k_range
        mask_k1 = ks1 < K
        a_ptrs1 = A_ptr + row_offs + ks1[None, :] * stride_ak
        b_ptrs1 = B_ptr + ks1 * stride_bk + 0 * stride_bn
        a_tile1 = tl.load(a_ptrs1, mask=mask_m[:, None] & mask_k1[None, :], other=0.0)
        b_tile1 = tl.load(b_ptrs1, mask=mask_k1, other=0.0)

        # Compute on the previous tile while next tile is prefetched
        acc += tl.sum(a_tile0.to(tl.float32) * b_tile0.to(tl.float32)[None, :], axis=1)

        # Rotate buffers
        a_tile0 = a_tile1
        b_tile0 = b_tile1
        k0 += BLOCK_K

    # Final tile compute
    acc += tl.sum(a_tile0.to(tl.float32) * b_tile0.to(tl.float32)[None, :], axis=1)

    # Store results to C[:, 0]
    c_ptrs = C_ptr + rows * stride_cm + 0 * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m)


class ModelNew(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    Optimized with a Triton kernel on CUDA devices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        # Use Triton kernel on CUDA with float32 inputs and expected shapes; otherwise fallback to torch.matmul
        use_triton = (
            A.is_cuda
            and B.is_cuda
            and A.dtype == torch.float32
            and B.dtype == torch.float32
            and A.ndim == 2
            and B.ndim == 2
            and B.shape[1] == 1
            and A.shape[1] == B.shape[0]
        )
        if not use_triton:
            return torch.matmul(A, B)

        M, K = A.shape
        # Ensure predictable strides and coalesced access
        A_ctg = A.contiguous()
        B_ctg = B.contiguous()
        C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

        # Heuristic tiling for large-K matvec on H200: 1 row per program, large K tile
        BLOCK_M = 1
        BLOCK_K = 8192  # 16 iterations for K=131072

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _gemv_rowblock_kernel[grid](
            A_ctg, B_ctg, C,
            M, K,
            A_ctg.stride(0), A_ctg.stride(1),
            B_ctg.stride(0), B_ctg.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=4,
        )
        return C


M = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed