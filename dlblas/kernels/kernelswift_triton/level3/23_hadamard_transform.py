import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-PyTorch Fast Walsh-Hadamard Transform (iterative butterfly)
# ---------------------------------------------------------------------------

def _hadamard_transform_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Iterative in-place Fast Walsh-Hadamard Transform.

    Args:
        x: (..., dim)  — dim need not be a power of 2; zero-padded internally.
        scale: scalar applied to the output.
    Returns:
        Tensor of shape (..., dim).
    """
    orig_n = x.shape[-1]
    log2_n = max(1, math.ceil(math.log2(orig_n)))
    n = 1 << log2_n  # next power of 2

    if orig_n < n:
        x = F.pad(x, (0, n - orig_n))

    orig_dtype = x.dtype
    result = x.to(torch.float32)
    prefix = result.shape[:-1]

    h = 1
    while h < n:
        # (..., n) -> (..., n//(2h), 2, h): expose butterfly pairs
        result = result.reshape(*prefix, n // (2 * h), 2, h)
        a = result[..., 0, :]   # upper butterfly input
        b = result[..., 1, :]   # lower butterfly input
        result = torch.stack([a + b, a - b], dim=-2)
        result = result.reshape(*prefix, n)
        h *= 2

    return (result[..., :orig_n] * scale).to(orig_dtype)


class _HadamardFn(torch.autograd.Function):
    """Autograd function so gradients flow through the transform."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        ctx._scale = scale
        return _hadamard_transform_impl(x, scale)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        # H is symmetric, so H^T == H.
        return _hadamard_transform_impl(dout, ctx._scale), None


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """Fast Walsh-Hadamard Transform as an nn.Module.

    Multiplies each row of the input by the (normalised or unnormalised)
    Hadamard matrix.  Equivalent to
        F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale
    If dim is not a power of 2, the input is zero-padded to the next power.

    Args:
        scale: scalar multiplied into the output (default 1.0).
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            out: (..., dim)
        """
        return _HadamardFn.apply(x, self.scale)


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

def get_init_inputs():
    """Positional args for Model.__init__: (scale,)."""
    return [1.0]


def get_inputs():
    """Positional args for Model.forward: (x,).

    x: (batch=4, dim=256) — float32, CUDA if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(4, 256, dtype=torch.float32, device=device)
    return [x]
