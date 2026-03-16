import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _layernorm_sums_kernel(
    x_ptr,         # (rows, M)
    sums_ptr,      # (rows,)
    sumsq_ptr,     # (rows,)
    M,             # int: number of features to normalize over
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    col_start = pid_col * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    row_start = pid_row * M
    idx = row_start + offsets

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    s = tl.sum(x, axis=0)
    s2 = tl.sum(x * x, axis=0)

    tl.atomic_add(sums_ptr + pid_row, s)
    tl.atomic_add(sumsq_ptr + pid_row, s2)


@triton.jit
def _layernorm_stats_kernel(
    sums_ptr,      # (rows,)
    sumsq_ptr,     # (rows,)
    mean_ptr,      # (rows,)
    rstd_ptr,      # (rows,)
    INV_M,         # float32 = 1.0 / M
    EPSILON,       # float32
):
    pid = tl.program_id(axis=0)
    s = tl.load(sums_ptr + pid).to(tl.float32)
    s2 = tl.load(sumsq_ptr + pid).to(tl.float32)
    mean = s * INV_M
    var = s2 * INV_M - mean * mean
    rstd = tl.rsqrt(var + EPSILON)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def _layernorm_apply_kernel(
    x_ptr,         # (rows, M)
    w_ptr,         # (M,)
    b_ptr,         # (M,)
    mean_ptr,      # (rows,)
    rstd_ptr,      # (rows,)
    y_ptr,         # (rows, M)
    M,             # int
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    col_start = pid_col * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    row_start = pid_row * M
    idx = row_start + offsets

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + pid_row)
    rstd = tl.load(rstd_ptr + pid_row)

    y = (x - mean) * rstd
    y = y * w + b
    tl.store(y_ptr + idx, y, mask=mask)


def _layer_norm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, normalized_shape: tuple, eps: float):
    # Flatten to (rows, M)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    M = math.prod(normalized_shape)
    rows = x.numel() // M

    x_in = x.contiguous().view(rows, M)
    y_out = torch.empty_like(x_in)

    w = weight.contiguous().view(M)
    b = bias.contiguous().view(M)

    device = x.device
    dtype_acc = torch.float32

    # Buffers for statistics
    sums = torch.zeros(rows, device=device, dtype=dtype_acc)
    sumsq = torch.zeros(rows, device=device, dtype=dtype_acc)
    mean = torch.empty(rows, device=device, dtype=dtype_acc)
    rstd = torch.empty(rows, device=device, dtype=dtype_acc)

    # Tile config
    def grid_2d(meta):
        return (rows, triton.cdiv(M, meta['BLOCK_SIZE']))

    # Reasonable default BLOCK_SIZE for large M; avoids unrolled loops
    BLOCK = 8192

    # Pass 1: accumulate sums and sumsq
    _layernorm_sums_kernel[grid_2d](x_in, sums, sumsq, M, BLOCK_SIZE=BLOCK)

    # Pass 2: compute mean and rstd per row
    inv_m = float(1.0 / M)
    _layernorm_stats_kernel[(rows,)](sums, sumsq, mean, rstd, inv_m, float(eps))

    # Pass 3: apply normalization + affine
    _layernorm_apply_kernel[grid_2d](x_in, w, b, mean, rstd, y_out, M, BLOCK_SIZE=BLOCK)

    return y_out.view_as(x)


class ModelNew(nn.Module):
    """
    LayerNorm implemented with a fast Triton kernel. Matches torch.nn.LayerNorm semantics.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.float32))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch on CPU for correctness
        if not x.is_cuda:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return _layer_norm_triton(x, self.weight, self.bias, self.normalized_shape, self.eps)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]