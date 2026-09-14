# Copyright (c) 2025, DeepLink.
"""Triton kernel implementation for vector_add.

Simple vector addition: c = a + b
"""

import torch
import triton
import triton.language as tl

# Define autotune configs as a module-level constant
# This ensures configs are available during torch.compile tracing
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=1),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N"])
@triton.jit
def vector_add_kernel(
    a_ptr,  # Pointer to first input vector
    b_ptr,  # Pointer to second input vector
    c_ptr,  # Pointer to output vector
    N,  # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program processes
):
    """Triton kernel for vector addition.

    Each program instance processes BLOCK_SIZE elements.
    """
    # Program ID
    pid = tl.program_id(axis=0)

    # Compute the range of elements this program will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle cases where N is not divisible by BLOCK_SIZE
    mask = offsets < N

    # Load inputs with boundary checking
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Compute addition
    c = a + b

    # Store output with boundary checking
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Main implementation function called by PyTorch op.

    Args:
        a: First input tensor (1D vector)
        b: Second input tensor (1D vector)

    Returns:
        c: Output tensor (a + b)

    This is the actual computation function that will be registered
    to torch.ops.dlblas.vector_add
    """
    # Parameter validation
    # Support both CUDA and NPU devices
    device_type = a.device.type
    if device_type not in ["cuda", "npu"]:
        raise RuntimeError(
            f"vector_add only supports CUDA or NPU tensors, got {device_type}"
        )

    if a.dim() != 1 or b.dim() != 1:
        raise RuntimeError(
            f"vector_add expects 1D tensors, got {a.dim()}D and {b.dim()}D"
        )

    if a.shape[0] != b.shape[0]:
        raise RuntimeError(f"vector length mismatch: {a.shape[0]} vs {b.shape[0]}")

    if a.dtype != b.dtype:
        raise RuntimeError(f"dtype mismatch: {a.dtype} vs {b.dtype}")

    N = a.shape[0]

    # Allocate output
    c = torch.empty_like(a)

    # Grid configuration: number of program instances needed
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

    # Launch kernel
    vector_add_kernel[grid](
        a,
        b,
        c,
        N,
    )

    return c
