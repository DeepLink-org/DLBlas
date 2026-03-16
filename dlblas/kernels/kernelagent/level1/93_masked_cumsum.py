import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cumsum_lastdim_kernel(
    x_ptr,      # *dtype
    out_ptr,    # *dtype
    M,          # int32
    N,          # int32
    stride_xm,  # int32
    stride_xn,  # int32
    stride_om,  # int32
    stride_on,  # int32
    BLOCK_N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Row base pointers
    row_x = x_ptr + pid_m * stride_xm
    row_o = out_ptr + pid_m * stride_om

    # Initialize running sum as typed zero (no global read occurs due to mask=False)
    running = tl.load(row_x, mask=False, other=0)

    # Process the row in BLOCK_N chunks
    for chunk in tl.static_range(NUM_CHUNKS):
        col_start = chunk * BLOCK_N

        # Pointers to the start of this chunk
        base_x = row_x + col_start * stride_xn
        base_o = row_o + col_start * stride_on

        # Incremental pointers to reduce address arithmetic
        px = base_x
        po = base_o

        # Sequential scan within the chunk from registers with strict bounds checks
        for i in tl.static_range(BLOCK_N):
            col = col_start + i
            in_bounds = col < N
            v = tl.load(px, mask=in_bounds, other=0)
            running = running + v
            tl.store(po, running, mask=in_bounds)
            px += stride_xn
            po += stride_on


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # CPU fallback
        if (not x.is_cuda) or (not mask.is_cuda):
            return torch.cumsum(x * mask, dim=self.dim)

        # Apply mask first to exactly match original semantics
        y = x * mask

        # Supported dtypes
        if y.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.cumsum(y, dim=self.dim)

        # Normalize dimension
        dim = self.dim if self.dim >= 0 else (self.dim + y.dim())
        N = y.size(dim)

        # Heuristic: PyTorch's native cumsum is very fast for large last-dim sizes.
        # Use Triton only for relatively smaller widths to amortize launch/scan overhead.
        if N > 2048:
            return torch.cumsum(y, dim=dim)

        # Move the target dim to the last dimension for a 2D [M, N] view
        if dim != y.dim() - 1:
            y = y.movedim(dim, -1)
        y = y.contiguous()

        # Collapse leading dims into M and use last dim as N
        M = y.numel() // y.size(-1)
        N = y.size(-1)
        y2d = y.view(M, N)
        out2d = torch.empty_like(y2d)

        # Strides for Triton
        stride_xm, stride_xn = y2d.stride()
        stride_om, stride_on = out2d.stride()

        # Tune tile size for good occupancy on Hopper
        if N <= 128:
            BLOCK_N = 128
        elif N <= 512:
            BLOCK_N = 256
        else:
            BLOCK_N = 512
        NUM_CHUNKS = (N + BLOCK_N - 1) // BLOCK_N

        grid = (M,)
        _cumsum_lastdim_kernel[grid](
            y2d,
            out2d,
            M,
            N,
            stride_xm,
            stride_xn,
            stride_om,
            stride_on,
            BLOCK_N=BLOCK_N,
            NUM_CHUNKS=NUM_CHUNKS,
            num_warps=2,
            num_stages=2,
        )

        out = out2d.view_as(y)
        # Move dim back to original position if needed
        if dim != x.dim() - 1:
            out = out.movedim(-1, dim)
        return out


batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape, device='cuda')
    mask = torch.randint(0, 2, x.shape, device=x.device).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]