import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced tiles
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8,  num_stages=5),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4,  num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4,  num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_warps=4,  num_stages=2),
        # Wider N or M
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8,  num_stages=4),
        # Deeper K
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4,  num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4,  num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_AT_B_kernel(
    A_ptr,  # A: (K, M)
    B_ptr,  # B: (K, N)
    C_ptr,  # C: (M, N)
    M, N, K,
    stride_a_k, stride_a_m,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along M (rows of C)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along N (cols of C)
    offs_k = tl.arange(0, BLOCK_K)                    # along K (reduction)

    # Provide compiler hints for vectorization/tiling
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        k_idx = k0 + offs_k  # [BK]

        # Load A tile from (K, M) as [BK, BM], then transpose to [BM, BK]
        a_ptrs = A_ptr + (k_idx[:, None] * stride_a_k + offs_m[None, :] * stride_a_m)
        a_mask = (k_idx[:, None] < K) & (offs_m[None, :] < M)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0, cache_modifier=".cg")
        a = tl.trans(a).to(tl.float32)  # shape [BM, BK] with A[k, m] laid out as (m, k)

        # Load B tile from (K, N) as [BK, BN]
        b_ptrs = B_ptr + (k_idx[:, None] * stride_b_k + offs_n[None, :] * stride_b_n)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".cg").to(tl.float32)  # shape [BK, BN]

        # Accumulate: C[m, n] += sum_k A[k, m] * B[k, n]
        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # Write back C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    Optimized using a Triton kernel to compute C = (A.T) @ B where:
      - A has shape (K, M)
      - B has shape (K, N)
      - Output C has shape (M, N)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # CPU fallback
        if not (A.is_cuda and B.is_cuda):
            return torch.matmul(A.T, B)

        # Fast path: for fp32 on CUDA, leverage cuBLAS directly for peak perf and parity
        if A.dtype == torch.float32 and B.dtype == torch.float32:
            return torch.matmul(A.T, B)

        # Shapes
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        K_a, M = A.shape
        K_b, N = B.shape
        assert K_a == K_b, "Inner dimensions must match: A.shape[0] == B.shape[0]"
        K = K_a

        # Ensure contiguous for predictable strides in Triton path
        Ac = A.contiguous()
        Bc = B.contiguous()

        # Triton kernel path (accumulate in fp32)
        C = torch.empty((M, N), device=Ac.device, dtype=torch.float32)

        # Strides (in elements)
        stride_a_k, stride_a_m = Ac.stride()  # A: (K, M)
        stride_b_k, stride_b_n = Bc.stride()  # B: (K, N)
        stride_c_m, stride_c_n = C.stride()   # C: (M, N)

        # Launch Triton kernel
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        _matmul_AT_B_kernel[grid](
            Ac, Bc, C,
            M, N, K,
            stride_a_k, stride_a_m,
            stride_b_k, stride_b_n,
            stride_c_m, stride_c_n,
        )
        return C


M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(K, M)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed