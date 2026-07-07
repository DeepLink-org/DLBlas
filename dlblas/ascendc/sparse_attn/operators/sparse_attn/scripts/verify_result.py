# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# SparseAttn 精度验证脚本
#
# 精度标准 (BF16 输出):
#   MERE < 2^-7 ≈ 0.00781
#   MARE < 10 × 2^-7 ≈ 0.0781
# ============================================================================

import numpy as np
import sys


def verify_result(output_path, golden_path, b, m, h, d):
    """Compare Ascend C output against golden.

    The output and golden are stored as bf16 (uint16 representation).
    """
    # Read binary files as uint16 (bf16 representation)
    output_uint16 = np.fromfile(output_path, dtype=np.uint16)
    golden_uint16 = np.fromfile(golden_path, dtype=np.uint16)

    total_elements = b * m * h * d
    if len(output_uint16) != total_elements:
        print(f"Shape mismatch: output has {len(output_uint16)} elements, "
              f"expected {total_elements} ({b}*{m}*{h}*{d})")
        return False
    if len(golden_uint16) != total_elements:
        print(f"Golden shape mismatch: golden has {len(golden_uint16)} elements, "
              f"expected {total_elements}")
        return False

    # Convert uint16 to float32 for comparison
    # bf16 → float32: view as uint16, extend to uint32, shift left 16 bits, view as float32
    output_uint32 = output_uint16.astype(np.uint32) << 16
    output_fp32 = output_uint32.view(np.float32)

    golden_uint32 = golden_uint16.astype(np.uint32) << 16
    golden_fp32 = golden_uint32.view(np.float32)

    # Compute error metrics
    abs_err = np.abs(output_fp32 - golden_fp32)

    # MERE: Maximum Element-wise Relative Error
    # Avoid division by zero
    denom = np.maximum(np.abs(golden_fp32), 1e-8)
    rel_err = abs_err / denom

    mere = np.max(rel_err)
    mare = np.mean(rel_err)

    max_abs = np.max(abs_err)
    mean_abs = np.mean(abs_err)

    # Thresholds
    mere_threshold = 2.0 ** -7   # ≈ 0.00781
    mare_threshold = 10.0 * mere_threshold  # ≈ 0.0781

    print(f"Shape: [{b}, {m}, {h}, {d}] ({total_elements} elements)")
    print(f"MERE: {mere:.8f}  (threshold: {mere_threshold:.8f})  {'PASS' if mere <= mere_threshold else 'FAIL'}")
    print(f"MARE: {mare:.8f}  (threshold: {mare_threshold:.8f})  {'PASS' if mare <= mare_threshold else 'FAIL'}")
    print(f"MaxAbsErr: {max_abs:.8f}")
    print(f"MeanAbsErr: {mean_abs:.8f}")

    # Also check for NaN/Inf
    nan_count = np.sum(np.isnan(output_fp32))
    inf_count = np.sum(np.isinf(output_fp32))
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in output!")
    if inf_count > 0:
        print(f"WARNING: {inf_count} Inf values in output!")

    passed = (mere <= mere_threshold) and (mare <= mare_threshold)
    if passed:
        print("Verification PASSED!")
    else:
        # Show worst elements
        worst_indices = np.argsort(rel_err)[-10:][::-1]
        print("Top 10 worst elements:")
        for idx in worst_indices:
            flat_idx = idx
            bi = flat_idx // (m * h * d)
            rem = flat_idx % (m * h * d)
            mi = rem // (h * d)
            rem = rem % (h * d)
            hi = rem // d
            di = rem % d
            print(f"  [{bi},{mi},{hi},{di}]: out={output_fp32[idx]:.8f} "
                  f"golden={golden_fp32[idx]:.8f} rel_err={rel_err[idx]:.8f}")
        print("Verification FAILED!")

    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin> [b m h d]")
        sys.exit(1)

    output_path = sys.argv[1]
    golden_path = sys.argv[2]

    b, m, h, d = 2, 16, 8, 64
    if len(sys.argv) >= 7:
        b = int(sys.argv[3])
        m = int(sys.argv[4])
        h = int(sys.argv[5])
        d = int(sys.argv[6])

    success = verify_result(output_path, golden_path, b, m, h, d)
    sys.exit(0 if success else 1)
