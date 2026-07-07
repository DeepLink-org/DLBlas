# Kernel 2: weights_projection
# This is a thin wrapper that re-exports from kernel_q_proj.py
# Both matmul kernels share the same underlying implementation.
# Separated for architectural clarity per the 4-kernel design.

from .kernel_q_proj import matmul_kernel, weights_projection

__all__ = ["weights_projection"]
