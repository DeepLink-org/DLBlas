# apply_mix precision verification with MERE/MARE metrics

import numpy as np
import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import bf16_uint16_to_fp32

# Precision thresholds per DESIGN.md (bf16 floating-point community standard)
MERE_THRESHOLD = 0.00781   # 2^-7
MARE_THRESHOLD = 0.0781    # 10 * 2^-7


def verify_result(output_path, golden_path, n0, n1, h):
    """Verify kernel output against golden reference.

    Args:
        output_path: path to kernel output binary (bf16 uint16)
        golden_path: path to golden binary (bf16 uint16)
        n0, n1, h: output shape [n0, n1, h]
    """
    total_elements = n0 * n1 * h

    # Read binary data as uint16 (bf16 representation)
    output_raw = np.fromfile(output_path, dtype=np.uint16)
    golden_raw = np.fromfile(golden_path, dtype=np.uint16)

    if len(output_raw) != total_elements:
        print(f"ERROR: Output size mismatch. Expected {total_elements}, got {len(output_raw)}")
        return False

    if len(golden_raw) != total_elements:
        print(f"ERROR: Golden size mismatch. Expected {total_elements}, got {len(golden_raw)}")
        return False

    # Convert bf16 → fp32
    output_fp32 = bf16_uint16_to_fp32(output_raw)
    golden_fp32 = bf16_uint16_to_fp32(golden_raw)

    # Compute errors
    abs_diff = np.abs(output_fp32 - golden_fp32)
    rel_diff = np.zeros_like(abs_diff)
    nonzero_mask = np.abs(golden_fp32) > 1e-30
    rel_diff[nonzero_mask] = abs_diff[nonzero_mask] / np.abs(golden_fp32[nonzero_mask])

    # Filter out cases where golden is zero (relative error undefined)
    mere = np.mean(rel_diff[nonzero_mask]) if np.any(nonzero_mask) else 0.0
    mare = np.max(rel_diff[nonzero_mask]) if np.any(nonzero_mask) else 0.0

    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    print(f"Verification results:")
    print(f"  Shape: [{n0}, {n1}, {h}] ({total_elements} elements)")
    print(f"  MERE (mean relative error): {mere:.6f} (threshold: {MERE_THRESHOLD:.6f})")
    print(f"  MARE (max relative error):  {mare:.6f} (threshold: {MARE_THRESHOLD:.6f})")
    print(f"  Max absolute difference:    {max_abs_diff:.6e}")
    print(f"  Mean absolute difference:   {mean_abs_diff:.6e}")

    mere_pass = mere < MERE_THRESHOLD
    mare_pass = mare < MARE_THRESHOLD

    if mere_pass and mare_pass:
        print(f"Verification PASSED!")
        return True
    else:
        if not mere_pass:
            print(f"MERE FAILED: {mere:.6f} >= {MERE_THRESHOLD:.6f}")
        if not mare_pass:
            print(f"MARE FAILED: {mare:.6f} >= {MARE_THRESHOLD:.6f}")

        # Show worst mismatches
        if np.any(nonzero_mask):
            worst_indices = np.argsort(rel_diff[nonzero_mask])[-10:]
            print(f"\nTop 10 worst relative errors:")
            for idx in reversed(worst_indices):
                full_idx = np.where(nonzero_mask)[0][idx]
                print(f"  [{full_idx}]: output={output_fp32[full_idx]:.6e}, "
                      f"golden={golden_fp32[full_idx]:.6e}, "
                      f"rel_err={rel_diff[full_idx]:.6e}")

        print(f"Verification FAILED!")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python verify_result.py <output.bin> <golden.bin> <n0> <n1> <h>")
        sys.exit(1)

    output_path = sys.argv[1]
    golden_path = sys.argv[2]
    n0 = int(sys.argv[3])
    n1 = int(sys.argv[4])
    h = int(sys.argv[5])

    success = verify_result(output_path, golden_path, n0, n1, h)
    sys.exit(0 if success else 1)
