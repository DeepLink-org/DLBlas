import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _row_scale_kernel(
    a_ptr,        # *A: shape [N]
    b_ptr,        # *B: shape [N, M]
    c_ptr,        # *C: shape [N, M]
    N, M,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    pid_n = tl.program_id(1)  # block id along columns

    row = pid_m
    col_start = pid_n * BLOCK_N
    offs_n = col_start + tl.arange(0, BLOCK_N)

    # Vectorization/alignment hints for better coalescing
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(offs_n, 8)
    tl.static_assert(BLOCK_N % 8 == 0)

    # In-bounds checks
    row_in_bounds = row < N
    cols_in_bounds = offs_n < M

    # Pointers for B and C tiles
    b_ptrs = b_ptr + row * stride_bm + offs_n * stride_bn
    c_ptrs = c_ptr + row * stride_cm + offs_n * stride_cn
    tl.multiple_of(b_ptrs, 16)
    tl.multiple_of(c_ptrs, 16)

    # Detect full interior tile along N-dimension to avoid predication
    full_n = (col_start + BLOCK_N) <= M
    full_tile = row_in_bounds & full_n

    if full_tile:
        # Unmasked fast path
        a_val = tl.load(a_ptr + row, cache_modifier=".ca")
        b = tl.load(b_ptrs, cache_modifier=".cg")
        tl.store(c_ptrs, b * a_val)
    else:
        # Boundary-safe masked path
        mask = row_in_bounds & cols_in_bounds
        a_val = tl.load(a_ptr + row, mask=row_in_bounds, other=0, cache_modifier=".ca")
        b = tl.load(b_ptrs, mask=mask, other=0, cache_modifier=".cg")
        tl.store(c_ptrs, b * a_val, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        # Fallback for non-CUDA tensors or shape mismatch
        if (not A.is_cuda) or (not B.is_cuda) or (A.dim() != 1) or (B.dim() != 2) or (A.shape[0] != B.shape[0]):
            return torch.diag(A) @ B

        N = A.shape[0]
        M = B.shape[1]

        # Match PyTorch matmul dtype promotion rules
        out_dtype = torch.result_type(A, B)
        A_cast = A.contiguous().to(out_dtype)
        B_cast = B.contiguous().to(out_dtype)
        C = torch.empty((N, M), device=B_cast.device, dtype=out_dtype)

        # Use a wider column tile for better bandwidth utilization
        BLOCK_N = 512
        grid = lambda meta: (N, triton.cdiv(M, meta['BLOCK_N']))
        _row_scale_kernel[grid](
            A_cast, B_cast, C,
            N, M,
            B_cast.stride(0), B_cast.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=1,
        )
        return C

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed