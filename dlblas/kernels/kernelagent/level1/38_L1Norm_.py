import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l1norm_row_kernel(
    x_ptr, y_ptr,
    B, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_in_bounds = pid < B

    # Hints for compiler: access is contiguous along columns
    tl.max_contiguous(cols, BLOCK_SIZE)
    tl.multiple_of(cols, 8)

    x_row_ptr = x_ptr + pid * stride_xm
    y_row_ptr = y_ptr + pid * stride_ym

    # Pass 1: compute L1 norm (sum of abs), 4x unrolled to reduce loop/control overhead
    acc = tl.zeros((), dtype=tl.float32)
    start = 0
    while start < N:
        offs0 = start + cols
        tl.multiple_of(offs0, 8)
        mask0 = (offs0 < N) & row_in_bounds
        x0 = tl.load(x_row_ptr + offs0 * stride_xn, mask=mask0, other=0.0, cache_modifier=".cg")
        acc += tl.sum(tl.abs(x0.to(tl.float32)), axis=0)

        offs1 = offs0 + BLOCK_SIZE
        tl.multiple_of(offs1, 8)
        mask1 = (offs1 < N) & row_in_bounds
        x1 = tl.load(x_row_ptr + offs1 * stride_xn, mask=mask1, other=0.0, cache_modifier=".cg")
        acc += tl.sum(tl.abs(x1.to(tl.float32)), axis=0)

        offs2 = offs1 + BLOCK_SIZE
        tl.multiple_of(offs2, 8)
        mask2 = (offs2 < N) & row_in_bounds
        x2 = tl.load(x_row_ptr + offs2 * stride_xn, mask=mask2, other=0.0, cache_modifier=".cg")
        acc += tl.sum(tl.abs(x2.to(tl.float32)), axis=0)

        offs3 = offs2 + BLOCK_SIZE
        tl.multiple_of(offs3, 8)
        mask3 = (offs3 < N) & row_in_bounds
        x3 = tl.load(x_row_ptr + offs3 * stride_xn, mask=mask3, other=0.0, cache_modifier=".cg")
        acc += tl.sum(tl.abs(x3.to(tl.float32)), axis=0)

        start += 4 * BLOCK_SIZE

    denom = acc
    inv = 1.0 / denom  # preserves division-by-zero semantics

    # Pass 2: normalize, 4x unrolled
    start = 0
    while start < N:
        offs0 = start + cols
        tl.multiple_of(offs0, 8)
        mask0 = (offs0 < N) & row_in_bounds
        x0 = tl.load(x_row_ptr + offs0 * stride_xn, mask=mask0, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(y_row_ptr + offs0 * stride_yn, x0 * inv, mask=mask0, eviction_policy="evict_last")

        offs1 = offs0 + BLOCK_SIZE
        tl.multiple_of(offs1, 8)
        mask1 = (offs1 < N) & row_in_bounds
        x1 = tl.load(x_row_ptr + offs1 * stride_xn, mask=mask1, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(y_row_ptr + offs1 * stride_yn, x1 * inv, mask=mask1, eviction_policy="evict_last")

        offs2 = offs1 + BLOCK_SIZE
        tl.multiple_of(offs2, 8)
        mask2 = (offs2 < N) & row_in_bounds
        x2 = tl.load(x_row_ptr + offs2 * stride_xn, mask=mask2, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(y_row_ptr + offs2 * stride_yn, x2 * inv, mask=mask2, eviction_policy="evict_last")

        offs3 = offs2 + BLOCK_SIZE
        tl.multiple_of(offs3, 8)
        mask3 = (offs3 < N) & row_in_bounds
        x3 = tl.load(x_row_ptr + offs3 * stride_xn, mask=mask3, other=0.0, cache_modifier=".cg").to(tl.float32)
        tl.store(y_row_ptr + offs3 * stride_yn, x3 * inv, mask=mask3, eviction_policy="evict_last")

        start += 4 * BLOCK_SIZE


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    Uses a Triton kernel on CUDA for 2D tensors; falls back to PyTorch otherwise.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton path: 2D CUDA tensor in common floating dtypes
        if x.is_cuda and x.ndim == 2 and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            B, N = x.shape
            # Ensure good memory access for inputs; outputs keep original layout
            x_contig = x.contiguous()
            y = torch.empty_like(x)

            stride_xm, stride_xn = x_contig.stride()
            stride_ym, stride_yn = y.stride()

            # Heuristic tiling: larger tiles for long rows to reduce passes
            if N >= 16384:
                BLOCK_SIZE = 4096
            elif N >= 8192:
                BLOCK_SIZE = 2048
            elif N >= 2048:
                BLOCK_SIZE = 1024
            else:
                # largest power-of-two <= N, ensure at least 64
                BLOCK_SIZE = max(64, (1 << (N.bit_length() - 1)) if N > 0 else 1)

            # Tune warps for throughput/occupancy balance
            if BLOCK_SIZE >= 2048:
                num_warps = 8
                num_stages = 6
            elif BLOCK_SIZE >= 1024:
                num_warps = 4
                num_stages = 5
            else:
                num_warps = 2 if BLOCK_SIZE < 512 else 4
                num_stages = 4

            _l1norm_row_kernel[(B,)](
                x_contig, y,
                B, N,
                stride_xm, stride_xn,
                stride_ym, stride_yn,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return y

        # Fallback: exact PyTorch semantics
        return x / torch.sum(torch.abs(x), dim=1, keepdim=True)


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []