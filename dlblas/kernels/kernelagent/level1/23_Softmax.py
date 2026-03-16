import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_row_fwd_db_kernel(
    x_ptr,   # *[B, D]
    y_ptr,   # *[B, D]
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(axis=0)
    row_start = pid * N_COLS
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)

    # -------- Pass 1: compute row-wise max and denominator with online update --------
    row_max = tl.full([1], -float("inf"), dtype=tl.float32)
    denom = tl.zeros([1], dtype=tl.float32)

    col = 0
    idx = row_start + col + offs
    mask = (col + offs) < N_COLS
    x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)

    while col < N_COLS:
        # Prefetch next tile
        col_next = col + BLOCK_SIZE
        idx_next = row_start + col_next + offs
        mask_next = (col_next + offs) < N_COLS
        x_next = tl.load(
            x_ptr + idx_next,
            mask=mask_next,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)

        # Online LSE combination
        tile_max = tl.max(x, axis=0)
        new_row_max = tl.maximum(row_max, tile_max)
        denom = denom * tl.exp(row_max - new_row_max) + tl.sum(tl.exp(x - new_row_max), axis=0)
        row_max = new_row_max

        # Advance
        col = col_next
        idx = idx_next
        x = x_next
        mask = mask_next

    inv_denom = 1.0 / denom

    # -------- Pass 2: write normalized probabilities --------
    col = 0
    idx = row_start + col + offs
    mask = (col + offs) < N_COLS
    x = tl.load(x_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    while col < N_COLS:
        # Prefetch next tile
        col_next = col + BLOCK_SIZE
        idx_next = row_start + col_next + offs
        mask_next = (col_next + offs) < N_COLS
        x_next = tl.load(
            x_ptr + idx_next,
            mask=mask_next,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        y = tl.exp(x - row_max) * inv_denom
        tl.store(y_ptr + idx, y, mask=mask)

        # Advance
        col = col_next
        idx = idx_next
        x = x_next
        mask = mask_next


def _select_kernel_config(n_cols: int):
    # Heuristic tuned for Hopper/H200: keep tiles moderate to reduce register pressure
    if n_cols >= 16384:
        return 2048, 8, 5
    if n_cols >= 8192:
        return 2048, 8, 4
    if n_cols >= 4096:
        return 1024, 4, 4
    if n_cols >= 2048:
        return 1024, 4, 3
    return 512, 4, 3


class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation using a Triton-optimized kernel when on CUDA.
    Falls back to torch.softmax for non-CUDA tensors or unsupported shapes.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if tensor is not 2D or not CUDA
        if x.dim() != 2 or not x.is_cuda:
            return torch.softmax(x, dim=1)

        # Ensure contiguous for coalesced access
        x_in = x.contiguous()
        B, D = x_in.shape
        y_out = torch.empty_like(x_in)

        BLOCK_SIZE, num_warps, num_stages = _select_kernel_config(D)
        grid = (B,)

        _softmax_row_fwd_db_kernel[grid](
            x_in, y_out,
            N_COLS=D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y_out


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed