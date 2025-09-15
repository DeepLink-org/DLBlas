
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
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pre-load weights and bias in fp32 to avoid repeated conversions
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    W_f32 = W_row.to(tl.float32)
    B_f32 = B_row.to(tl.float32)

    # Calculate pointers for this row
    row_X_ptr = X_ptr + row_idx * X_row_stride
    row_Y_ptr = Y_ptr + row_idx * Y_row_stride
    row_Mean_ptr = Mean_ptr + row_idx * Mean_row_stride
    row_RSTD_ptr = RSTD_ptr + row_idx * RSTD_row_stride

    # Load input data and convert to fp32 for numerical stability
    X_row = tl.load(row_X_ptr + col_offsets, mask=mask, other=0.0)
    X_f32 = X_row.to(tl.float32)

    # Compute statistics in fp32 for numerical stability
    n_cols_f32 = n_cols.to(tl.float32)
    mean = tl.sum(X_f32, axis=0) / n_cols_f32
    X_centered = X_f32 - mean
    # Apply mask to variance calculation to exclude contributions from masked elements
    X_centered_masked = tl.where(mask, X_centered, 0.0)
    var = tl.sum(X_centered_masked * X_centered_masked, axis=0) / n_cols_f32
    rstd = rsqrt(var + eps)
    # Store statistics (convert back to original dtype only once)
    tl.store(row_Mean_ptr, mean.to(X_row.dtype))
    tl.store(row_RSTD_ptr, rstd.to(X_row.dtype))
    # Fused normalization and affine transformation
    # Y = (X - mean) * rstd * W + B = X_centered * rstd * W + B
    Y_f32 = X_centered * rstd * W_f32 + B_f32

    # Store output (single conversion back to original dtype)
    tl.store(row_Y_ptr + col_offsets, Y_f32.to(X_row.dtype), mask=mask)

def call(X, W, B, eps):
    """
    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability
    Returns:
        Tuple of (output, input, mean, rstd, block_size, num_warps)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )
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
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
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
