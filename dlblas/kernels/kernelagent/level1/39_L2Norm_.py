import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l2norm_rowwise_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return

    # Base pointers for this row
    row_x_ptr = x_ptr + pid * stride_xm
    row_y_ptr = y_ptr + pid * stride_ym

    cols = tl.arange(0, BLOCK_N)
    col_offs_x = cols * stride_xn
    col_offs_y = cols * stride_yn

    # First pass: compute L2 norm of the row in fp32
    sumsq = tl.zeros([1], dtype=tl.float32)
    n = 0
    while n < N:
        offs = n + cols
        mask = offs < N
        x = tl.load(row_x_ptr + (n * stride_xn) + col_offs_x, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sumsq += tl.sum(xf * xf, axis=0)
        n += BLOCK_N

    # Use reciprocal sqrt to reduce div latency; 0 -> inf, which yields NaN for 0*inf as in PyTorch (0/0)
    inv_norm = tl.rsqrt(sumsq)

    # Second pass: write normalized values
    n = 0
    while n < N:
        offs = n + cols
        mask = offs < N
        x = tl.load(row_x_ptr + (n * stride_xn) + col_offs_x, mask=mask, other=0.0)
        y = x * inv_norm
        tl.store(row_y_ptr + (n * stride_yn) + col_offs_y, y, mask=mask)
        n += BLOCK_N


def _select_block_and_warps(N: int):
    # Choose a power-of-two block size for good memory coalescing
    if N >= 16384:
        block = 4096
    elif N >= 8192:
        block = 2048
    elif N >= 4096:
        block = 1024
    elif N >= 2048:
        block = 512
    elif N >= 1024:
        block = 256
    else:
        block = 128
    # Tune warps for the chosen block size
    if block >= 4096:
        warps = 8
    elif block >= 1024:
        warps = 4
    else:
        warps = 2
    return block, warps


class ModelNew(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        # Fallback to PyTorch for non-2D or non-CUDA tensors to preserve semantics
        if (not x.is_cuda) or (x.dim() != 2):
            return x / torch.norm(x, p=2, dim=1, keepdim=True)

        x_c = x.contiguous()
        B, D = x_c.shape
        y = torch.empty_like(x_c)

        stride_xm, stride_xn = x_c.stride()
        stride_ym, stride_yn = y.stride()

        BLOCK_N, num_warps = _select_block_and_warps(D)
        grid = (B,)

        _l2norm_rowwise_kernel[grid](
            x_c, y,
            B, D,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=4,
        )
        return y


batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []