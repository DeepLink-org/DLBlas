# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Result verification for engram_fused_weight
#
# Output: FP32 (float32 binary); Golden: FP32 (float32 binary)
# Precision standard per DESIGN.md §8.1 (FP32 output):
#   MERE (mean relative error) < 2^-13 ≈ 0.000122
#   MARE (max relative error)  < 10 * 2^-13 ≈ 0.00122
# ============================================================================

import numpy as np
import sys


def verify_result(output_path, golden_path):
    """Verify AscendC FP32 output against PyTorch FP32 golden.

    Returns True if precision thresholds are met.
    """
    output = np.fromfile(output_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if output.size == 0:
        print("WARNING: empty output and golden")
        return True

    # Check for Inf/NaN patterns
    output_finite = np.isfinite(output)
    golden_finite = np.isfinite(golden)
    if not np.array_equal(output_finite, golden_finite):
        n_diff = np.sum(output_finite != golden_finite)
        print(f"WARNING: output and golden have different Inf/NaN patterns ({n_diff} elements)")

    diff = np.abs(output - golden)
    max_diff = float(np.max(diff))

    # Per DESIGN.md §8.1: FP32 output precision thresholds
    mere_threshold = 2.0**(-13)          # ≈ 0.000122
    mare_threshold = 10.0 * 2.0**(-13)   # ≈ 0.00122

    # MERE/MARE: only over non-zero, finite golden values
    finite_mask = np.isfinite(golden) & (np.abs(golden) > 0)
    if np.any(finite_mask):
        rel_err = diff[finite_mask] / np.abs(golden[finite_mask])
        mere = float(np.mean(rel_err))
        mare = float(np.max(rel_err))
    else:
        mere = 0.0
        mare = 0.0

    mere_passed = mere < mere_threshold
    mare_passed = mare < mare_threshold
    passed = mere_passed and mare_passed

    print(f"Verification {'PASSED!' if passed else 'FAILED!'}")
    print(f"  Shape: {output.shape}, elements: {output.size}")
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  MERE (mean rel err): {mere:.6e}  (threshold: {mere_threshold:.6e}) {'PASS' if mere_passed else 'FAIL'}")
    print(f"  MARE (max rel err):  {mare:.6e}  (threshold: {mare_threshold:.6e}) {'PASS' if mare_passed else 'FAIL'}")

    if not passed:
        mismatches = np.where(diff > 0)[0]
        print(f"  Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            print(f"  First 5 mismatches at indices: {mismatches[:5]}")
            for idx in mismatches[:3]:
                print(f"    [{idx}] output={output[idx]:.6e} golden={golden[idx]:.6e} diff={diff[idx]:.6e}")

    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
