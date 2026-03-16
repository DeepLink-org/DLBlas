import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        # Additional configs to better utilize H200
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
    ],
    key=["N"],
)
@triton.jit
def _upper_tri_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_Am, stride_Ak,
    stride_Bk, stride_Bn,
    stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Out-of-range tiles can be skipped early
    if (m0 >= N) or (n0 >= N):
        return
    # If the whole tile is strictly below the diagonal, we can skip it.
    if m0 > (n0 + BLOCK_N - 1):
        return

    rm = m0 + tl.arange(0, BLOCK_M)
    rn = n0 + tl.arange(0, BLOCK_N)
    m_in = rm < N
    n_in = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    rk = tl.arange(0, BLOCK_K)
    tl.multiple_of(rm, BLOCK_M)
    tl.multiple_of(rn, BLOCK_N)
    tl.multiple_of(rk, BLOCK_K)

    # K sweep using tiled dot-product.
    # Keep inputs in their native dtype to leverage tensor cores for fp16/bf16,
    # while accumulating in fp32. Disable TF32 to match PyTorch fp32 numerics.
    for k0 in range(0, N, BLOCK_K):
        k = k0 + rk
        k_in = k < N

        a_ptrs = A_ptr + (rm[:, None] * stride_Am + k[None, :] * stride_Ak)
        b_ptrs = B_ptr + (k[:, None] * stride_Bk + rn[None, :] * stride_Bn)

        a = tl.load(a_ptrs, mask=m_in[:, None] & k_in[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_in[:, None] & n_in[None, :], other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C_ptr + (rm[:, None] * stride_Cm + rn[None, :] * stride_Cn)

    # Store only upper-triangular region
    tile_all_upper = (m0 + BLOCK_M - 1) <= n0
    full_in_bounds = (m0 + BLOCK_M) <= N and (n0 + BLOCK_N) <= N

    if tile_all_upper and full_in_bounds:
        # Fast path: no masks needed
        tl.store(c_ptrs, acc)
    else:
        store_mask = (rm[:, None] <= rn[None, :]) & m_in[:, None] & n_in[None, :]
        tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) and returns its upper triangular part.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication and returns torch.triu(A @ B).

        Args:
            A (torch.Tensor): Matrix of shape (N, N).
            B (torch.Tensor): Matrix of shape (N, N).

        Returns:
            torch.Tensor: Upper triangular part of A @ B, shape (N, N).
        """
        # Fallback to PyTorch if not on CUDA or shapes not 2D square
        if (A.device.type != "cuda") or (B.device.type != "cuda") or (A.dim() != 2) or (B.dim() != 2) or (A.shape != B.shape) or (A.shape[0] != A.shape[1]):
            return torch.triu(torch.matmul(A, B))

        N = A.shape[0]
        out_dtype = torch.result_type(A, B)

        # Ensure contiguous inputs and consistent dtype (matches PyTorch promotion rules)
        A_ = A.to(out_dtype).contiguous()
        B_ = B.to(out_dtype).contiguous()

        # Allocate output as zeros so we can skip strictly-below-diagonal tiles safely
        C = torch.zeros((N, N), device=A.device, dtype=out_dtype)

        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _upper_tri_matmul_kernel[grid](
            A_, B_, C,
            N,
            A_.stride(0), A_.stride(1),
            B_.stride(0), B_.stride(1),
            C.stride(0), C.stride(1),
        )
        # C already contains only the upper-triangular values
        return C


N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []