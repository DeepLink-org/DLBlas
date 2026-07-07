# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Result verification for head_compute_mix_bwd (3 outputs)
# ============================================================================

import numpy as np
import sys

dtype = np.float32
rtol = 1e-4
atol = 1e-6


def verify_one(name, output_path, golden_path, expected_shape=None):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    print(f"\n--- {name} ---")
    print(f"  output shape: {output.shape}, golden shape: {golden.shape}")

    if expected_shape is not None:
        expected_flat = np.prod(expected_shape)
        if output.shape[0] != expected_flat:
            print(f"  WARNING: output size {output.shape[0]} != expected {expected_flat}")

    if output.shape != golden.shape:
        print(f"  Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(output - golden))
        print(f"  PASSED! Max diff: {max_diff:.6e}")
        return True
    else:
        diff = np.abs(output - golden)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  FAILED! Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"  Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            print(f"  First few mismatches:")
            for idx in mismatches[:5]:
                print(f"    [{idx}] output={output[idx]:.8f}, golden={golden[idx]:.8f}, diff={diff[idx]:.6e}")
        return False


if __name__ == "__main__":
    # Shape: batch0=2, batch1=1024, mhc_mult=4
    total_rows = 2 * 1024
    inner_dim = 4

    results = []
    results.append(verify_one(
        "grad_input_mix",
        "output/output_grad_input_mix.bin",
        "output/golden_grad_input_mix.bin",
        (2, 1024, 4)))

    results.append(verify_one(
        "grad_mhc_scale",
        "output/output_grad_mhc_scale.bin",
        "output/golden_grad_mhc_scale.bin",
        (1,)))

    results.append(verify_one(
        "grad_mhc_base",
        "output/output_grad_mhc_base.bin",
        "output/golden_grad_mhc_base.bin",
        (4,)))

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*50}")
    print(f"Summary: {passed}/{total} outputs passed")
    print(f"Status: {'PASSED' if passed == total else 'FAILED'}")
    sys.exit(0 if passed == total else 1)
