import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _rcumsum_lastdim_kernel(
    x_ptr, y_ptr,
    rows, N,
    stride_x_row, stride_x_col,
    stride_y_row, stride_y_col,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    # Row base pointers
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    # Lane indices within a tile
    i = tl.arange(0, BLOCK_N)

    # Accumulator that propagates across tiles from right to left
    carry = tl.zeros((), dtype=tl.float32)

    # Prefetch the rightmost tile
    last_block = NUM_BLOCKS - 1
    base = last_block * BLOCK_N
    rev_cols = base + (BLOCK_N - 1 - i)
    m_rev = rev_cols < N
    x_rev = tl.load(row_x_ptr + rev_cols * stride_x_col, mask=m_rev, other=0.0).to(tl.float32)

    # Process tiles from rightmost to leftmost with simple software pipelining
    for b in range(NUM_BLOCKS):
        # Compute for the prefetched tile
        block_idx = NUM_BLOCKS - 1 - b
        base_cur = block_idx * BLOCK_N
        rev_cols_cur = base_cur + (BLOCK_N - 1 - i)
        m_rev_cur = rev_cols_cur < N

        # Inclusive scan within the reversed tile -> reverse-cumsum on original
        scan_rev = tl.cumsum(x_rev, axis=0)
        out_rev = scan_rev + carry
        tl.store(row_y_ptr + rev_cols_cur * stride_y_col, out_rev.to(tl.float32), mask=m_rev_cur)

        # Update carry with sum of this tile
        carry += tl.sum(x_rev, axis=0)

        # Prefetch next tile if any
        next_block_idx = block_idx - 1
        if next_block_idx >= 0:
            base_next = next_block_idx * BLOCK_N
            rev_cols_next = base_next + (BLOCK_N - 1 - i)
            m_rev_next = rev_cols_next < N
            x_rev = tl.load(row_x_ptr + rev_cols_next * stride_x_col, mask=m_rev_next, other=0.0).to(tl.float32)


class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # Fallback to PyTorch if input is not CUDA tensor
        if not x.is_cuda:
            return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

        dim = self.dim if self.dim >= 0 else (x.ndim + self.dim)
        assert 0 <= dim < x.ndim, "Invalid dim"

        # Move the target dimension to the last for coalesced access
        if dim != x.ndim - 1:
            perm = list(range(x.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            x_perm = x.permute(perm).contiguous()
            inv_perm = [0] * x.ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
        else:
            x_perm = x.contiguous()
            inv_perm = None

        # Flatten to [rows, N] with last dim contiguous
        N = x_perm.shape[-1]
        rows = x_perm.numel() // N
        x2 = x_perm.view(rows, N)
        y2 = torch.empty_like(x2)

        # Tiling configuration
        BLOCK_N = 512
        NUM_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
        grid = (rows,)

        _rcumsum_lastdim_kernel[grid](
            x2, y2,
            rows, N,
            x2.stride(0), x2.stride(1),
            y2.stride(0), y2.stride(1),
            NUM_BLOCKS=NUM_BLOCKS,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=4,
        )

        y_perm = y2.view_as(x_perm)
        if inv_perm is not None:
            y_out = y_perm.permute(inv_perm)
        else:
            y_out = y_perm
        return y_out


batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    # Prefer CUDA to exercise the custom Triton kernel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [torch.randn(batch_size, *input_shape, device=device)]

def get_init_inputs():
    return [dim]