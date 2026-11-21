# Copyright (c) 2025, DeepLink.
"""PyTorch operator registration module.

This module registers dlBLAS operators as PyTorch native operators,
enabling calls via torch.ops.dlblas.xxx()

All operators in this module follow PyTorch's torch.library registration mechanism,
making them compatible with:
- torch.compile
- torch.jit.trace / torch.jit.script
- PyTorch autograd (when implemented)
"""

import torch

# Check PyTorch version and library availability
try:
    from torch.library import Library

    HAS_TORCH_LIBRARY = True
except ImportError:
    HAS_TORCH_LIBRARY = False
    print(
        "Warning: torch.library not available (requires PyTorch 2.0+). "
        "PyTorch ops registration disabled."
    )

if HAS_TORCH_LIBRARY:
    # Create a library handle for dlblas operators
    # "FRAGMENT" allows incremental registration across modules
    _lib = Library("dlblas", "FRAGMENT")

    # Import and register all operators
    # Each operator module will use the shared _lib handle
    from . import vector_add  # noqa: F401

    __all__ = ["vector_add"]
