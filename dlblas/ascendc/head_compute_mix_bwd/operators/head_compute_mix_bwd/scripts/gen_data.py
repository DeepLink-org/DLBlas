# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Test data generation for head_compute_mix_bwd
#
# Usage:
#   python3 gen_data.py                  # Default shape (2, 1024, 4)
#   python3 gen_data.py 1 1 4            # Minimal shape
#   python3 gen_data.py 4 512 4          # Custom shape (n0, n1, mhc_mult)
#   python3 gen_data.py 2 4096 4         # Large n1 (ub_loops > 1)
# ============================================================================

import numpy as np
import os
import sys

from golden import compute_golden


def generate(n0, n1, mhc_mult):
    """Generate test data and golden outputs for given shape."""
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Generate input data
    np.random.seed(42)
    input_mix = np.random.randn(n0, n1, mhc_mult).astype(np.float32)
    mhc_scale = np.random.randn(1).astype(np.float32)
    mhc_base = np.random.randn(mhc_mult).astype(np.float32)
    grad_out = np.random.randn(n0, n1, mhc_mult).astype(np.float32)

    # Write inputs to binary files
    input_mix.tofile("input/input_input_mix.bin")
    mhc_scale.tofile("input/input_mhc_scale.bin")
    # For the kernel, mhc_base needs to be 8 elements (4 original + 4 repeated for broadcast alignment)
    mhc_base_8 = np.concatenate([mhc_base, mhc_base]).astype(np.float32)
    mhc_base_8.tofile("input/input_mhc_base.bin")
    grad_out.tofile("input/input_grad_out.bin")

    # Compute golden outputs
    grad_input_mix, grad_mhc_scale, grad_mhc_base = compute_golden(
        input_mix, mhc_scale, mhc_base, grad_out)

    # Write golden outputs
    grad_input_mix.tofile("output/golden_grad_input_mix.bin")
    grad_mhc_scale.tofile("output/golden_grad_mhc_scale.bin")
    grad_mhc_base.tofile("output/golden_grad_mhc_base.bin")

    print(f"Generated test data (n0={n0}, n1={n1}, mhc_mult={mhc_mult}):")
    print(f"  input_mix: {input_mix.shape}, {input_mix.dtype}")
    print(f"  mhc_scale: {mhc_scale.shape}, {mhc_scale.dtype}")
    print(f"  mhc_base:  {mhc_base.shape}, {mhc_base.dtype}")
    print(f"  grad_out:  {grad_out.shape}, {grad_out.dtype}")
    print(f"  Golden outputs:")
    print(f"    grad_input_mix:  {grad_input_mix.shape}")
    print(f"    grad_mhc_scale:  {grad_mhc_scale.shape}")
    print(f"    grad_mhc_base:   {grad_mhc_base.shape}")

    return n0, n1, mhc_mult


if __name__ == "__main__":
    n0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    mhc_mult = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    generate(n0, n1, mhc_mult)
