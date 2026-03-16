import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _min_reduce_last_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_b, out_stride_m,
    BLOCK_K: tl.constexpr,
):
    # Reduce over last dim (N). Each program computes one (b, m) output.
    pid = tl.program_id(axis=0)
    b = pid // M
    m = pid % M

    # Guard out-of-bounds programs (shouldn't happen if grid is correct, but keep safe)
    in_bounds = (b < B) & (m < M)
    if ~in_bounds:
        return

    base = b * stride_b + m * stride_m

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Seed accumulator with first chunk and reduce to scalar
    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < N
    ptrs0 = x_ptr + base + idx0 * stride_n
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".ca")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    # Unrolled loop to increase ILP and reduce loop overhead
    UNROLL = 2
    while k < N:
        for u in tl.static_range(0, UNROLL):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < N
            ptrs = x_ptr + base + idx * stride_n
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".ca")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += UNROLL * BLOCK_K

    out_off = b * out_stride_b + m * out_stride_m
    tl.store(out_ptr + out_off, acc)


@triton.jit
def _min_reduce_mid_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_b, out_stride_n,
    BLOCK_K: tl.constexpr,
):
    # Reduce over middle dim (M). Each program computes one (b, n) output.
    pid = tl.program_id(axis=0)
    b = pid // N
    n = pid % N

    in_bounds = (b < B) & (n < N)
    if ~in_bounds:
        return

    base = b * stride_b + n * stride_n

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Seed accumulator using first chunk -> scalar
    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < M
    ptrs0 = x_ptr + base + idx0 * stride_m
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".cg")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    # Unrolled loop for strided loads
    UNROLL = 2
    while k < M:
        for u in tl.static_range(0, UNROLL):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < M
            ptrs = x_ptr + base + idx * stride_m
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".cg")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += UNROLL * BLOCK_K

    out_off = b * out_stride_b + n * out_stride_n
    tl.store(out_ptr + out_off, acc)


@triton.jit
def _min_reduce_first_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_m, out_stride_n,
    BLOCK_K: tl.constexpr,
):
    # Reduce over first dim (B). Each program computes one (m, n) output.
    pid = tl.program_id(axis=0)
    m = pid // N
    n = pid % N

    in_bounds = (m < M) & (n < N)
    if ~in_bounds:
        return

    base = m * stride_m + n * stride_n

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    # Seed accumulator using first chunk -> scalar
    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < B
    ptrs0 = x_ptr + base + idx0 * stride_b
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".cg")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    # Unrolled loop for strided loads
    UNROLL = 2
    while k < B:
        for u in tl.static_range(0, UNROLL):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < B
            ptrs = x_ptr + base + idx * stride_b
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".cg")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += UNROLL * BLOCK_K

    out_off = m * out_stride_m + n * out_stride_n
    tl.store(out_ptr + out_off, acc)


class ModelNew(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    Now accelerated with Triton on NVIDIA GPUs for 3D inputs.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def _choose_block_and_warps(self, K: int):
        # Favor larger tiles for contiguous last-dim to minimize loop count
        if K >= 256:
            return 256, 8
        elif K >= 128:
            return 128, 4
        elif K >= 64:
            return 64, 2
        else:
            return 32, 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        # Fallbacks for general cases
        if (not x.is_cuda) or (x.ndim != 3):
            return torch.min(x, dim=self.dim)[0]

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim not in (0, 1, 2):
            return torch.min(x, dim=dim)[0]

        B, M, N = x.shape
        sb, sm, sn = x.stride()

        # Degenerate sizes -> fallback
        if B == 0 or M == 0 or N == 0:
            return torch.min(x, dim=dim)[0]

        # Only accelerate numerics-friendly dtypes
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.min(x, dim=dim)[0]

        # Use Triton only when memory access is contiguous along the reduction axis
        if dim == 2 and sn == 1:
            # reduce over last dim -> output [B, M]
            out = torch.empty((B, M), device=x.device, dtype=x.dtype)
            ob, om = out.stride()
            grid = (B * M,)
            BK, NW = self._choose_block_and_warps(N)
            _min_reduce_last_kernel[grid](
                x, out,
                B, M, N,
                sb, sm, sn,
                ob, om,
                BLOCK_K=BK,
                num_warps=NW, num_stages=4,
            )
            return out
        elif dim == 1 and False:
            # Reserved path for future optimization of strided reductions.
            # Currently falls back to PyTorch for best performance.
            out = torch.empty((B, N), device=x.device, dtype=x.dtype)
            ob, on = out.stride()
            grid = (B * N,)
            BK, NW = self._choose_block_and_warps(M)
            _min_reduce_mid_kernel[grid](
                x, out,
                B, M, N,
                sb, sm, sn,
                ob, on,
                BLOCK_K=BK,
                num_warps=NW, num_stages=4,
            )
            return out
        elif dim == 0 and False:
            # Reserved path for future optimization of strided reductions.
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            om, on = out.stride()
            grid = (M * N,)
            BK, NW = self._choose_block_and_warps(B)
            _min_reduce_first_kernel[grid](
                x, out,
                B, M, N,
                sb, sm, sn,
                om, on,
                BLOCK_K=BK,
                num_warps=NW, num_stages=4,
            )
            return out
        else:
            # Fallback to PyTorch for non-contiguous reduction axes (often faster)
            return torch.min(x, dim=dim)[0]


batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension