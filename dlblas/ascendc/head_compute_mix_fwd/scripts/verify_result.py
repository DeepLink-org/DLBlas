# ============================================================================
# Result verification for head_compute_mix_fwd
#
# Usage:
#   python3 verify_result.py output/output.bin output/golden.bin
# ============================================================================

import numpy as np
import sys

dtype = np.float16
# Relaxed tolerances for FP16 sigmoid computation
rtol = 1e-2
atol = 1e-3


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    # Use FP32 for comparison to catch precision issues
    output_f32 = output.astype(np.float32)
    golden_f32 = golden.astype(np.float32)

    diff = np.abs(output_f32 - golden_f32)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Check allclose
    passed = np.allclose(output_f32, golden_f32, rtol=rtol, atol=atol)

    if passed:
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        return True
    else:
        print(f"Verification FAILED!")
        print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden_f32))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            # Show first few mismatches
            show_n = min(5, len(mismatches))
            for i in range(show_n):
                idx = mismatches[i]
                print(f"  [{idx}]: output={output_f32[idx]:.8f}, golden={golden_f32[idx]:.8f}, "
                      f"diff={diff[idx]:.6e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
