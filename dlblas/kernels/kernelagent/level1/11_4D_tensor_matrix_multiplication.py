import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for rows/cols within the tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointer for the C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Accumulator in fp32 (numerically stable, matches PyTorch semantics)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop along K dimension, blocked by BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # Build pointers for current A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_range[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_range[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary checks
        a_mask = (offs_m[:, None] < M) & (k_range[None, :] < K)
        b_mask = (k_range[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles and upcast to fp32 for robust numerical agreement across dtypes
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Blocked matmul accumulate with strict FP32 math (disable TF32)
        acc += tl.dot(a, b, allow_tf32=False)

    # Write back results with boundary mask; pointer dtype of C determines cast
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _matmul_triton(A2d: torch.Tensor, B: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Computes C = A2d @ B using a Triton kernel.
    A2d: (M, K)
    B:   (K, N)
    Returns C: (M, N) in dtype=out_dtype
    """
    assert A2d.dim() == 2 and B.dim() == 2
    M, K = A2d.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimensions must match for matmul"

    A_ptr = A2d
    B_ptr = B

    # Allocate output
    C = torch.empty((M, N), device=A2d.device, dtype=out_dtype)

    # Strides in units of elements
    stride_am, stride_ak = A2d.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Launch grid with tiles
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _matmul_2d_kernel[grid](
        A_ptr, B_ptr, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Shapes: A(b, i, j, l), B(l, k) -> C(b, i, j, k)
        assert A.dim() == 4 and B.dim() == 2, "Invalid input ranks"
        b, i, j, l = A.shape
        lb, k = B.shape
        assert l == lb, "Mismatched reduction dimension"

        # Use Triton kernel on CUDA for supported dtypes; otherwise fallback to PyTorch
        if A.is_cuda and B.is_cuda and (A.dtype in (torch.float16, torch.bfloat16, torch.float32)) and (B.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            # Flatten (b, i, j, l) -> (b*i*j, l)
            A2d = A.reshape(-1, l)
            # Result dtype follows PyTorch semantics
            out_dtype = torch.result_type(A, B)
            # Triton matmul: (b*i*j, l) @ (l, k) -> (b*i*j, k)
            C2d = _matmul_triton(A2d, B, out_dtype)
            # Reshape to (b, i, j, k)
            C = C2d.view(b, i, j, k)
            return C
        else:
            # CPU or non-supported dtype/device fallback preserving exact semantics
            return torch.einsum("bijl,lk->bijk", A, B)


# Test code
b = 16
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = torch.randn(b, i, j, l)
    B = torch.randn(l, k)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed