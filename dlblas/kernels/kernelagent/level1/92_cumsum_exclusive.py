import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _exclusive_cumsum_row_to_padded_kernel(
    x_ptr,          # *const T, [B, N], row-major contiguous
    y_ptr,          # *mut T,   [B-1, N+1], row-major contiguous
    n_rows_out,     # B - 1
    n_cols_in,      # N
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_valid = pid < n_rows_out

    # Base offsets for this row
    row_x_base = pid * n_cols_in
    row_y_base = pid * (n_cols_in + 1)

    # Write the leading zero column
    tl.store(y_ptr + row_y_base, 0, mask=row_valid)

    idx = tl.arange(0, BLOCK_SIZE)
    valid = row_valid & (idx < n_cols_in)

    # Offsets
    offs_x = row_x_base + idx
    offs_y = row_y_base + 1 + idx  # shifted by +1 to account for the prepended zero

    # Load input row segment
    x = tl.load(x_ptr + offs_x, mask=valid, other=0)

    # Inclusive scan using in-place y_ptr as scratch
    running = x
    tl.store(y_ptr + offs_y, running, mask=valid)

    step = 1
    while step < BLOCK_SIZE:
        shifted = tl.load(y_ptr + (offs_y - step), mask=valid & (idx >= step), other=0)
        running = running + shifted
        tl.store(y_ptr + offs_y, running, mask=valid)
        step *= 2


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Fast path: CUDA 2D tensor, dim == 1 (or -1), supported dtype, and N > 0
        if (
            x.is_cuda
            and x.ndim == 2
            and (self.dim % x.ndim) == 1
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.shape[1] > 0
        ):
            B, N = x.shape
            # Output matches exact original semantics:
            # exclusive_cumsum = cat([zeros_like(first-slice along dim), x], dim) -> [B, N+1]
            # then "[:-1]" slices along dim=0 -> [B-1, N+1]
            if B <= 1:
                return x.new_empty((0, N + 1))

            y = torch.empty((B - 1, N + 1), device=x.device, dtype=x.dtype)

            # BLOCK_SIZE as next power-of-two >= N (cap to 4096)
            p2 = 1 << (N - 1).bit_length()
            BLOCK = min(p2, 4096)

            grid = (B - 1,)
            _exclusive_cumsum_row_to_padded_kernel[grid](
                x, y,
                B - 1, N,
                BLOCK_SIZE=BLOCK,
                num_warps=8 if BLOCK >= 1024 else 4,
                num_stages=4,
            )
            return y

        # Fallback for generic shapes/dtypes/devices: exact original behavior
        exclusive_cumsum = torch.cat(
            (torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x),
            dim=self.dim
        )[:-1]
        return torch.cumsum(exclusive_cumsum, dim=self.dim)


batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    # Use CUDA to exercise the Triton kernel during evaluation
    return [torch.randn(batch_size, *input_shape, device='cuda')]

def get_init_inputs():
    return [dim]