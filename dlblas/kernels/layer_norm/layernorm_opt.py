
import pytest
import torch
import triton
import triton.language as tl

from dlblas.utils.device_utils import infer_device

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.math import rsqrt

@triton.jit
def next_multiple_of(a, b):
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_N': BN
        }, num_stages=s, num_warps=w) for BN in [2048, 4096] for s in [2,3,4] for w in [4, 8]
    ],
    key=['n_cols'],
)
@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    pid_m = tl.program_id(0)
    n_cols_f32 = n_cols.to(tl.float32)
    sum = tl.zeros((1,), dtype=tl.float32)
    prev_multiple = prev_multiple_of(n_cols, BLOCK_N)
    # 第一次循环，非尾块，不带mask
    for start_n in range(0, prev_multiple, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        offset = pid_m * n_cols + n_offsets
        X_ptrs = X_ptr + offset
        x = tl.load(X_ptrs).to(tl.float32)
        sum = tl.sum(x, axis=0) + sum

    # 第一次循环的尾块，带mask
    n_offsets = prev_multiple + tl.arange(0, BLOCK_N)
    offset = pid_m * n_cols + n_offsets
    X_ptrs = X_ptr + offset
    mask = n_offsets < n_cols
    x = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum = tl.sum(x, axis=0) + sum
    
    
    mean = sum / n_cols_f32
    var = tl.zeros((1,), dtype=tl.float32)
    # 第二次循环最右侧块（同上一次循环的尾块），带mask
    n_offsets = prev_multiple + tl.arange(0, BLOCK_N)
    offset = pid_m * n_cols + n_offsets
    X_ptrs = X_ptr + offset
    mask = n_offsets < n_cols
    x = tl.load(X_ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    x_centered = x - mean
    x_centered_masked = tl.where(mask, x_centered, 0.0)
    var = tl.sum(x_centered_masked * x_centered_masked, axis=0) + var
    # 第二次循环的非尾块，不带mask
    for start_n in range(BLOCK_N, n_cols, BLOCK_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, BLOCK_N)
        offset = pid_m * n_cols + n_offsets
        X_ptrs = X_ptr + offset
        x = tl.load(X_ptrs, eviction_policy="evict_first").to(tl.float32)
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) + var
    # 第三次循环的非尾块，不带mask
    var = var / n_cols_f32
    for start_n in range(0, prev_multiple, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        offset = pid_m * n_cols + n_offsets
        x = tl.load(X_ptr + offset).to(tl.float32)
        w = tl.load(W_ptr + n_offsets).to(tl.float32)
        b = tl.load(B_ptr + n_offsets).to(tl.float32)
        rstd = rsqrt(var + eps)
        x_centered = x - mean
        y = (x_centered * rstd * w + b).to(tl.bfloat16)
        tl.store(Y_ptr + offset, y)

    n_offsets = prev_multiple + tl.arange(0, BLOCK_N)
    offset = pid_m * n_cols + n_offsets
    mask = n_offsets < n_cols
    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = rsqrt(var + eps)
    x_centered = x - mean
    y = (x_centered * rstd * w + b).to(tl.bfloat16)
    tl.store(Y_ptr + offset, y, mask=mask)


def call(X, W, B, eps):
    """
    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability
    Returns:
        Tuple of (output, input, mean, rstd, BLOCK_N, num_warps)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    # Allocate output tensors
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # Validate input dimensions
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # Launch kernel with one thread block per row for optimal performance
    grid = (n_rows,)
    _layer_norm_forward_kernel[grid](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        B,
        B.stride(0),
        n_cols,
        eps,
    )
    return Y.view(*shape)



def bench_fn(X, W, B, eps):
    fn = lambda: call(X, W, B, eps)
    ms = triton.testing.do_bench(fn, warmup=100, rep=100)
    return ms


def register(op_name):
    from dlblas.utils import SymVar, Tensor, register_dlblas_op
    for dtype in [torch.bfloat16]:
        for device in ['cuda']:
            seq_len = SymVar('seq_len') 
            hidden_size = SymVar('hidden_size')
            # we dont' actually allocate tensor
            x = Tensor((seq_len, hidden_size), dtype=dtype, device=device)
            w = Tensor((hidden_size,), dtype=dtype, device=device)
            b = Tensor((hidden_size,), dtype=dtype, device=device)
            register_dlblas_op(op_name, None, (x, w, b, torch.SymFloat), call, bench_fn, call)
