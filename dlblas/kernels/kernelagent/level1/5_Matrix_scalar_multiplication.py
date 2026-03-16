import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x_ptr, y_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Fast path for full tiles: avoid mask overhead entirely
    full = block_start + BLOCK_SIZE <= n_elements
    if full:
        x = tl.load(x_ptr + offsets, cache_modifier=".cg")
        s_cast = tl.full((), s, x.dtype)
        y = x * s_cast
        tl.store(y_ptr + offsets, y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        s_cast = tl.full((), s, x.dtype)
        y = x * s_cast
        tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        # Fallback to PyTorch on CPU to preserve original behavior
        if not A.is_cuda:
            return A * s

        A = A.contiguous()
        C = torch.empty_like(A)
        n_elements = A.numel()
        if n_elements == 0:
            return C

        # 1D grid over the flattened tensor
        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        # Use a block size that evenly divides common large shapes to take the full-tile path
        _scale_kernel[grid](
            A, C, float(s), n_elements,
            BLOCK_SIZE=16384,
            num_warps=8,
            num_stages=1,
        )
        return C


M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed