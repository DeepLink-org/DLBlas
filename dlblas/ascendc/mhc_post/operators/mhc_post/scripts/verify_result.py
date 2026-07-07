# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 结果验证脚本 - MHC Post (bf16)
# 对比 kernel 输出与 golden 参考 (bf16，使用 MERE/MARE 精度标准)
# bf16 精度标准 (DESIGN.md §9.2):
#   MERE < 2^-7 ≈ 7.81e-3
#   MARE < 10 × 2^-7 ≈ 7.81e-2
# ============================================================================

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import bf16_to_fp32


def read_bf16_bin(path, shape):
    """Read bf16 binary file and convert to fp32 for comparison."""
    data = np.fromfile(path, dtype=np.uint16)
    return bf16_to_fp32(data).reshape(shape)


def verify_result(output_path, golden_path, shape):
    """Verify bf16 kernel output against golden using MERE/MARE standards."""
    output = read_bf16_bin(output_path, shape)
    golden = read_bf16_bin(golden_path, shape)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    diff = np.abs(output - golden)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    total = int(np.prod(golden.shape))
    golden_abs = np.abs(golden) + 1e-7

    # Relative error per element
    rel_err = diff / golden_abs

    # MERE = Mean Element-wise Relative Error
    mere = np.mean(rel_err)

    # MARE = Max Absolute Relative Error
    # For bf16 near-zero values (|golden| < atol_gate), skip from MARE
    # since relative error is not meaningful for values near bf16 precision floor.
    # bf16 mantissa = 7 bits, 1 ULP at 1e-7 ≈ 5.96e-8
    atol_gate = 1e-5  # values below this are near bf16 noise floor
    valid_mask = golden_abs > (atol_gate + 1e-7)
    mare = np.max(rel_err[valid_mask]) if np.any(valid_mask) else 0.0

    # Count near-zero elements excluded from MARE
    n_near_zero = int(np.sum(~valid_mask))

    # bf16 precision thresholds (DESIGN.md §9.2)
    mere_threshold = 2.0 ** -7       # ≈ 7.81e-3
    mare_threshold = 10.0 * (2.0 ** -7)  # ≈ 7.81e-2

    mere_pass = mere < mere_threshold
    mare_pass = mare < mare_threshold

    print(f"=== MHC Post Verification (bf16) ===")
    print(f"Shape: {golden.shape}, Total elements: {total}")
    print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if n_near_zero > 0:
        print(f"Near-zero elements excluded from MARE (|golden| < {atol_gate:.0e}): {n_near_zero}")
    print(f"MERE ({mere:.6e}) < threshold ({mere_threshold:.6e}): {'PASS' if mere_pass else 'FAIL'}")
    print(f"MARE ({mare:.6e}) < threshold ({mare_threshold:.6e}): {'PASS' if mare_pass else 'FAIL'}")

    if mere_pass and mare_pass:
        print(f"Verification PASSED!")
        return True
    else:
        print(f"Verification FAILED!")
        # Show top error locations (excluding near-zero elements from MARE)
        worst_indices = np.argsort(rel_err.ravel())[-10:][::-1]
        worst_indices_unraveled = np.unravel_index(worst_indices, rel_err.shape)
        print("Top 10 worst relative errors:")
        for idx in zip(*worst_indices_unraveled):
            near_zero_flag = " [near-zero]" if golden_abs[idx] <= (atol_gate + 1e-7) else ""
            print(f"  [{idx}] output={output[idx]:.6e} golden={golden[idx]:.6e} "
                  f"diff={diff[idx]:.6e} rel_err={rel_err[idx]:.6e}{near_zero_flag}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin> "
              "[n0] [n1] [h] [mhc_mult]")
        sys.exit(1)

    output_path = sys.argv[1]
    golden_path = sys.argv[2]

    if len(sys.argv) >= 7:
        n0, n1, h, mhc_mult = map(int, sys.argv[3:7])
        shape = (n0, n1, mhc_mult, h)
    else:
        shape = (2, 4096, 4, 1280)

    success = verify_result(output_path, golden_path, shape)
    sys.exit(0 if success else 1)
