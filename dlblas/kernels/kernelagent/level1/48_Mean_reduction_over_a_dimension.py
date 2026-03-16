import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Reduce over last dimension (dim=2): x[b, m, :] -> y[b, m]
@triton.jit
def _mean_reduce_last_kernel(
    x_ptr, y_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    y_stride_b, y_stride_m,
    invN,  # float32
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // M
    m = pid % M
    if (b >= B) | (m >= M):
        return

    base_ptr = x_ptr + b * stride_b + m * stride_m
    offs = tl.arange(0, BLOCK_N)

    # Accumulate across N in a vector register; reduce once at the end
    acc_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    k = 0
    UNROLL: tl.constexpr = 4
    while k < N:
        # Software unrolling for better ILP
        for u in tl.static_range(UNROLL):
            idx = k + u * BLOCK_N + offs
            mask = idx < N
            vals = tl.load(base_ptr + idx * stride_n, mask=mask, other=0.0)
            acc_vec += vals.to(tl.float32)
        k += UNROLL * BLOCK_N

    total = tl.sum(acc_vec, axis=0)
    mean = total * invN
    out_ptr = y_ptr + b * y_stride_b + m * y_stride_m
    tl.store(out_ptr, mean)


# Reduce over middle dimension (dim=1): x[b, :, n] -> y[b, n]
# Tile along contiguous N to keep loads coalesced.
@triton.jit
def _mean_reduce_mid_tiled_kernel(
    x_ptr, y_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    y_stride_b, y_stride_n,
    invM,  # float32
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    if b >= B:
        return

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    m = 0
    UNROLL: tl.constexpr = 4
    # Unroll over M to reduce loop overhead; guard with mask for tail
    while m < M:
        for u in tl.static_range(UNROLL):
            mi = m + u
            mi_valid = mi < M
            ptr = x_ptr + b * stride_b + mi * stride_m + offs_n * stride_n
            vals = tl.load(ptr, mask=n_mask & mi_valid, other=0.0).to(tl.float32)
            acc += vals
        m += UNROLL

    mean = acc * invM
    out_ptr = y_ptr + b * y_stride_b + offs_n * y_stride_n
    tl.store(out_ptr, mean, mask=n_mask)


# Reduce over first dimension (dim=0): x[:, m, n] -> y[m, n]
# Tile along contiguous N to keep loads coalesced.
@triton.jit
def _mean_reduce_first_tiled_kernel(
    x_ptr, y_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    y_stride_m, y_stride_n,
    invB,  # float32
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    if m >= M:
        return

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    b = 0
    UNROLL: tl.constexpr = 4
    # Unroll over B to improve ILP; guard tail with mask
    while b < B:
        for u in tl.static_range(UNROLL):
            bi = b + u
            bi_valid = bi < B
            ptr = x_ptr + bi * stride_b + m * stride_m + offs_n * stride_n
            vals = tl.load(ptr, mask=n_mask & bi_valid, other=0.0).to(tl.float32)
            acc += vals
        b += UNROLL

    mean = acc * invB
    out_ptr = y_ptr + m * y_stride_m + offs_n * y_stride_n
    tl.store(out_ptr, mean, mask=n_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        # Fallback to PyTorch for non-CUDA or unsupported shapes/dtypes to ensure exact semantics
        if not x.is_cuda:
            return torch.mean(x, dim=self.dim)

        x = x.contiguous()
        nd = x.dim()
        dim = self.dim if self.dim >= 0 else (self.dim + nd)
        if nd != 3 or not (0 <= dim < 3):
            return torch.mean(x, dim=self.dim)

        # Handle empty reduction dimensions (PyTorch returns nan)
        B, M, N = x.shape
        if (dim == 0 and B == 0) or (dim == 1 and M == 0) or (dim == 2 and N == 0):
            return torch.mean(x, dim=self.dim)

        # Only optimize fp32; fallback otherwise to keep dtype semantics exact
        if x.dtype != torch.float32:
            return torch.mean(x, dim=self.dim)

        device = x.device
        dtype = x.dtype

        # Choose tile along the contiguous N dimension for coalesced loads
        if dim == 2:
            # Reduce along last dimension: per (b, m)
            y = torch.empty((B, M), device=device, dtype=dtype)
            # Prefer larger tile to improve bandwidth utilization on Hopper
            BLOCK_N = 512 if N >= 512 else (256 if N >= 256 else (128 if N >= 128 else 64))
            grid = (B * M,)
            _mean_reduce_last_kernel[grid](
                x, y,
                B, M, N,
                x.stride(0), x.stride(1), x.stride(2),
                y.stride(0), y.stride(1),
                1.0 / float(N),
                BLOCK_N=BLOCK_N,
                num_warps=8 if BLOCK_N >= 256 else 4,
                num_stages=4,
            )
            return y
        elif dim == 1:
            # Reduce along middle dimension: tile across N for coalesced access
            y = torch.empty((B, N), device=device, dtype=dtype)
            BLOCK_N = 256 if N >= 256 else 128 if N >= 128 else 64
            grid = (B, triton.cdiv(N, BLOCK_N))
            _mean_reduce_mid_tiled_kernel[grid](
                x, y,
                B, M, N,
                x.stride(0), x.stride(1), x.stride(2),
                y.stride(0), y.stride(1),
                1.0 / float(M),
                BLOCK_N=BLOCK_N,
                num_warps=8 if BLOCK_N >= 256 else 4,
                num_stages=4,
            )
            return y
        else:
            # dim == 0: reduce along first dimension, tile across N for coalesced access
            y = torch.empty((M, N), device=device, dtype=dtype)
            BLOCK_N = 256 if N >= 256 else 128 if N >= 128 else 64
            grid = (M, triton.cdiv(N, BLOCK_N))
            _mean_reduce_first_tiled_kernel[grid](
                x, y,
                B, M, N,
                x.stride(0), x.stride(1), x.stride(2),
                y.stride(0), y.stride(1),
                1.0 / float(B),
                BLOCK_N=BLOCK_N,
                num_warps=8 if BLOCK_N >= 256 else 4,
                num_stages=4,
            )
            return y


batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]