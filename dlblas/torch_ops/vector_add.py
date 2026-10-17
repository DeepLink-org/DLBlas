# Copyright (c) 2025, DeepLink.
"""Register vector_add as a PyTorch operator.

This module registers the vector_add Triton kernel as a native PyTorch operator,
enabling calls via:
- torch.ops.dlblas.vector_add(a, b)
- Support for torch.compile
- Support for torch.jit tracing
"""

import torch
from torch.library import Library
from typing import Tuple

# Import the Triton kernel implementation
from dlblas.kernels.vector_add import vector_add_impl

# Get the shared library handle from parent module
from dlblas.torch_ops import _lib

# ===== Step 1: Define Meta Function =====


def vector_add_meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Meta function for vector_add - computes output shape/dtype without computation.

    This is used by PyTorch for:
    - Shape inference (for torch.compile, JIT tracing)
    - DType inference
    - Device inference

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Empty tensor with correct shape/dtype/device (no actual computation)

    Raises:
        RuntimeError: If inputs are not on CUDA or have mismatched shapes
    """
    # Basic validation
    if a.dim() != 1 or b.dim() != 1:
        raise RuntimeError(
            f"vector_add expects 1D tensors, got {a.dim()}D and {b.dim()}D"
        )

    if a.shape[0] != b.shape[0]:
        raise RuntimeError(f"vector length mismatch: {a.shape[0]} vs {b.shape[0]}")

    if a.dtype != b.dtype:
        raise RuntimeError(f"dtype mismatch: {a.dtype} vs {b.dtype}")

    # Return empty tensor with the correct metadata
    return torch.empty_like(a)


# ===== Step 2: Register Operator Schema =====

# Define the operator signature (schema)
_lib.define(
    "vector_add(Tensor a, Tensor b) -> Tensor",
)


# ===== Step 3: Register Implementations =====

# Register implementation for PrivateUse1 device (NPU on Ascend)
# Ascend NPU uses "PrivateUse1" as device key in PyTorch
_lib.impl("vector_add", vector_add_impl, "PrivateUse1")

# Also register for CUDA if available
_lib.impl("vector_add", vector_add_impl, "CUDA")

# Register Meta implementation (for shape inference)
_lib.impl("vector_add", vector_add_meta, "Meta")


# ===== Step 4: Optional - Register CPU Fallback =====


def vector_add_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CPU fallback implementation using PyTorch native operations.

    This allows the operator to work on CPU tensors as well.
    """
    return a + b


_lib.impl("vector_add", vector_add_cpu, "CPU")


# ===== Step 5: Optional - Autograd Support =====


def vector_add_backward(
    a: torch.Tensor,
    b: torch.Tensor,
    grad_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for vector_add.

    For c = a + b, the gradients are:
    - grad_a = grad_output
    - grad_b = grad_output

    Args:
        a: Original input a (for shape reference)
        b: Original input b (for shape reference)
        grad_output: Gradient of loss with respect to output c

    Returns:
        Tuple of (grad_a, grad_b)
    """
    return grad_output, grad_output


class VectorAddFunction(torch.autograd.Function):
    """Custom autograd function for vector_add with gradient support."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Save inputs for backward pass
        ctx.save_for_backward(a, b)
        return vector_add_impl(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        return vector_add_backward(a, b, grad_output)


def vector_add_with_autograd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector add with autograd support.

    Use this when you need gradient computation.
    Note: For CUDA tensors, this uses the Triton kernel in forward pass.
    """
    return VectorAddFunction.apply(a, b)


# ===== Convenience exports =====

__all__ = [
    "vector_add_impl",
    "vector_add_meta",
    "vector_add_cpu",
    "vector_add_with_autograd",
]
