import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _max_reduce_dim1_kernel(
    x_ptr, o_ptr,
    B, M, N,
    sx0, sx1, sx2,
    so0, so1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Grid: (B, ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Hints for vectorization along N
    tl.multiple_of(n_offsets, 16)
    tl.max_contiguous(n_offsets, BLOCK_N)

    base = x_ptr + b * sx0

    # Initialize accumulator in the same dtype as output without touching memory
    o_ptrs = o_ptr + b * so0 + n_offsets * so1
    acc = tl.load(o_ptrs, mask=tl.zeros([BLOCK_N], dtype=tl.int1), other=-float("inf"))

    # Stream across M rows, unrolled by BLOCK_M, and update running max
    m_start = 0
    while m_start < M:
        for mi in tl.static_range(0, BLOCK_M):
            m_idx = m_start + mi
            valid_row = m_idx < M
            row_ptrs = base + m_idx * sx1 + n_offsets * sx2
            row = tl.load(row_ptrs, mask=n_mask & valid_row, other=-float("inf"), cache_modifier=".cg")
            acc = tl.maximum(acc, row)
        m_start += BLOCK_M

    tl.store(o_ptrs, acc, mask=n_mask)


@triton.jit
def _max_reduce_dim0_kernel(
    x_ptr, o_ptr,
    B, M, N,
    sx0, sx1, sx2,
    so0, so1,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Grid: (M, ceil_div(N, BLOCK_N))
    m = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Double-buffered streaming across B
    b_start = 0
    b_offsets = b_start + tl.arange(0, BLOCK_B)
    b_mask0 = b_offsets < B
    ptrs0 = x_ptr + b_offsets[:, None] * sx0 + m * sx1 + n_offsets[None, :] * sx2
    mask0 = b_mask0[:, None] & n_mask[None, :]
    x_cur = tl.load(ptrs0, mask=mask0, other=-float("inf"))

    acc = tl.max(x_cur, axis=0)

    b_start += BLOCK_B
    while b_start < B:
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offsets < B
        ptrs = x_ptr + b_offsets[:, None] * sx0 + m * sx1 + n_offsets[None, :] * sx2
        mask = b_mask[:, None] & n_mask[None, :]
        x_next = tl.load(ptrs, mask=mask, other=-float("inf"))

        tile_max = tl.max(x_cur, axis=0)
        acc = tl.maximum(acc, tile_max)
        x_cur = x_next
        b_start += BLOCK_B

    tile_max = tl.max(x_cur, axis=0)
    acc = tl.maximum(acc, tile_max)

    o_ptrs = o_ptr + m * so0 + n_offsets * so1
    tl.store(o_ptrs, acc, mask=n_mask)


@triton.jit
def _max_reduce_dim2_kernel(
    x_ptr, o_ptr,
    B, M, N,
    sx0, sx1, sx2,
    so0, so1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Grid: (B, ceil_div(M, BLOCK_M))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    # Single-buffered streaming across N
    n_start = 0
    # Initialize accumulator in output dtype without reading memory
    o_ptrs = o_ptr + b * so0 + m_offsets * so1
    acc = tl.load(o_ptrs, mask=tl.zeros([BLOCK_M], dtype=tl.int1), other=-float("inf"))

    while n_start < N:
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        ptrs = x_ptr + b * sx0 + m_offsets[:, None] * sx1 + n_offsets[None, :] * sx2
        mask = m_mask[:, None] & n_mask[None, :]
        x_tile = tl.load(ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg")

        tile_max = tl.max(x_tile, axis=1)
        acc = tl.maximum(acc, tile_max)
        n_start += BLOCK_N

    tl.store(o_ptrs, acc, mask=m_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
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
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        # Fallback for non-CUDA, non-3D, or unsupported dtypes
        if (not x.is_cuda) or (x.dim() != 3) or (x.dtype not in (torch.float16, torch.float32, torch.bfloat16)):
            return torch.max(x, dim=self.dim)[0]

        B, M, N = x.shape
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if dim not in (0, 1, 2):
            return torch.max(x, dim=self.dim)[0]

        # Ensure contiguous for predictable strides
        x_c = x.contiguous()
        sx0, sx1, sx2 = x_c.stride()

        if dim == 1:
            # Reduce over M -> output [B, N]
            out = torch.empty((B, N), device=x.device, dtype=x.dtype)
            so0, so1 = out.stride()
            # Larger tiles along M to reduce loop iters; balanced N for occupancy
            BLOCK_M, BLOCK_N = 128, 128
            grid = (B, triton.cdiv(N, BLOCK_N))
            _max_reduce_dim1_kernel[grid](
                x_c, out,
                B, M, N,
                sx0, sx1, sx2,
                so0, so1,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=8, num_stages=4
            )
            return out
        elif dim == 0:
            # Reduce over B -> output [M, N]
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            so0, so1 = out.stride()
            # Match small batch; keep N wide for coalescing
            BLOCK_B, BLOCK_N = 16, 128
            grid = (M, triton.cdiv(N, BLOCK_N))
            _max_reduce_dim0_kernel[grid](
                x_c, out,
                B, M, N,
                sx0, sx1, sx2,
                so0, so1,
                BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N,
                num_warps=4, num_stages=4
            )
            return out
        else:
            # dim == 2: Reduce over N -> output [B, M]
            out = torch.empty((B, M), device=x.device, dtype=x.dtype)
            so0, so1 = out.stride()
            BLOCK_M, BLOCK_N = 128, 128
            grid = (B, triton.cdiv(M, BLOCK_M))
            _max_reduce_dim2_kernel[grid](
                x_c, out,
                B, M, N,
                sx0, sx1, sx2,
                so0, so1,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=8, num_stages=4
            )
            return out


batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension