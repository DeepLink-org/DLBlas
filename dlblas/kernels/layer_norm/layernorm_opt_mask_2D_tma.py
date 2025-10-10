
import pytest
import torch
import triton
import triton.language as tl
from dlblas.kernels.grouped_gemm.BF16.utils import TmaAutoTuneHelper
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
            'XBLOCK_SIZE': BX, 'RBLOCK_SIZE': BR
        }, num_stages=s, num_warps=w) for BX in [1024, 2048, 4096]  for BR in [8] for s in [2,3,4] for w in [4]
    ],
    key=['N', 'M'],
)
@triton.jit
def _layer_norm_forward_kernel(
    x_desc_ptr, y_desc_ptr,
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    stride,  # how much to increase the pointer when moving by 1 row
    N,
    M,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    XBLOCK_SIZE: tl.constexpr,
    RBLOCK_SIZE: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row_begin = tl.program_id(0) * RBLOCK_SIZE
    row_idx = row_begin + tl.arange(0,RBLOCK_SIZE)
    row_mask = row_idx < M
    row_offsets = row_idx[:,None]*stride
    # Compute mean
    _mean = tl.zeros((RBLOCK_SIZE, XBLOCK_SIZE), dtype=tl.float32)
    prev_multiple = prev_multiple_of(N, XBLOCK_SIZE)
    # 第一次循环的非尾块，使用tma方式加载 x
    for start_n in range(0, prev_multiple, XBLOCK_SIZE):
        col_idx = start_n + tl.arange(0, XBLOCK_SIZE)
        x = tl._experimental_descriptor_load(x_desc_ptr, [row_begin, start_n], [RBLOCK_SIZE, XBLOCK_SIZE], tl.bfloat16)
        _mean += x.to(tl.float32)
    # 第一次循环的尾块，使用 tl.load + mask 方式加载 x
    col_idx = prev_multiple + tl.arange(0, XBLOCK_SIZE)
    col_mask = col_idx < N
    mask = row_mask[:,None] & col_mask[None,:]
    a = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.0).to(tl.float32)
    _mean += a

    mean = tl.sum(_mean, axis=1, keep_dims = True) / N
    # Compute variance
    _var = tl.zeros((RBLOCK_SIZE, XBLOCK_SIZE), dtype=tl.float32)
    prev_multiple = prev_multiple_of(N, XBLOCK_SIZE)
    # 第二次循环的尾块，使用 tl.load + mask 方式加载 x
    col_idx = prev_multiple + tl.arange(0, XBLOCK_SIZE)
    col_mask = col_idx < N
    mask = row_mask[:,None] & col_mask[None,:]
    X_ptrs = X + row_offsets + col_idx[None,:]
    x = tl.load(X_ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    x = tl.where(mask, x - mean, 0.0)
    _var += x * x
    # 第二次循环的非尾块，使用tma方式加载 x
    for start_n in range(XBLOCK_SIZE, N, XBLOCK_SIZE):
        col_idx = (prev_multiple - start_n) + tl.arange(0, XBLOCK_SIZE)
        x = tl._experimental_descriptor_load(x_desc_ptr, [row_begin, prev_multiple - start_n], [RBLOCK_SIZE, XBLOCK_SIZE], tl.bfloat16)
        x = x.to(tl.float32)
        x = x - mean
        _var += x * x

    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = rsqrt(var + eps)
    prev_multiple = prev_multiple_of(N, XBLOCK_SIZE)
    # 第三次循环的非尾块，使用tma方式加载和写出
    for start_n in range(0, prev_multiple, XBLOCK_SIZE):
        col_idx = start_n + tl.arange(0, XBLOCK_SIZE)
        x = tl._experimental_descriptor_load(x_desc_ptr, [row_begin, start_n], [RBLOCK_SIZE, XBLOCK_SIZE], tl.bfloat16)
        x = x.to(tl.float32)
        w = tl.load(W + col_idx).reshape((1,XBLOCK_SIZE))
        b = tl.load(B + col_idx).reshape((1,XBLOCK_SIZE))
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl._experimental_descriptor_store(y_desc_ptr, y.to(tl.bfloat16), [row_begin, start_n])

    # 第三次循环的尾块，使用load/store + mask方式加载和写回
    col_idx = prev_multiple + tl.arange(0, XBLOCK_SIZE)
    col_mask = col_idx < N
    mask = row_mask[:,None] & col_mask[None,:]
    w = tl.load(W + col_idx, mask=col_mask).reshape((1,XBLOCK_SIZE))
    b = tl.load(B + col_idx, mask=col_mask).reshape((1,XBLOCK_SIZE))
    x = tl.load(X + row_offsets + col_idx[None,:], mask=mask, other=0.).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    # Write output
    tl.store(Y + row_offsets + col_idx[None,:], y.to(tl.bfloat16), mask=mask)


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
    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("x")
    desc_helper.init_tma_descriptor("y")
    def grid(META):
        assert (n_cols * X.element_size()) % 16 == 0, "TMA required 16-byte alignment"
        nonlocal desc_helper
        
        desc_helper.fill_2d_tma_descriptor(
            "x",
            X.data_ptr(),
            n_rows,
            n_cols,
            META['RBLOCK_SIZE'],
            META['XBLOCK_SIZE'],
            X.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "y",
            Y.data_ptr(),
            n_rows,
            n_cols,
            META['RBLOCK_SIZE'],
            META['XBLOCK_SIZE'],
            Y.element_size(),
        )

        return (triton.cdiv(n_rows, META['RBLOCK_SIZE']),)

    desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
    desc_y = desc_helper.get_tma_descriptor_kernel_param("y")
    _layer_norm_forward_kernel[grid](
        desc_x, desc_y,
        X,
        Y,
        W,
        B,
        X.stride(0),
        n_cols,
        n_rows,
        eps,
    )
    # print(f"best config {_layer_norm_forward_kernel.best_config}", flush = True)
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
