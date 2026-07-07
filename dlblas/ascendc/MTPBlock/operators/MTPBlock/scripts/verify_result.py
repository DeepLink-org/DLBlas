# ============================================================================
# MTPBlock 精度验证脚本
# ============================================================================

import numpy as np
import sys

# Default: fp16 (bf16 compatible)
# Usage: python verify_result.py <output.bin> <golden.bin> [dtype=fp16|fp32]
dtype_map = {'fp16': np.float16, 'fp32': np.float32}


def verify_result(output_path, golden_path, dtype_str='fp16'):
    dtype = dtype_map.get(dtype_str, np.float16)
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    # 转为 fp32 做更精确的误差计算
    out_f32 = output.astype(np.float32)
    gold_f32 = golden.astype(np.float32)

    abs_diff = np.abs(out_f32 - gold_f32)
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)

    # MERE (Maximum Element-wise Relative Error)
    # MARE (Maximum Average Relative Error)
    denom = np.maximum(np.abs(gold_f32), 1e-8)
    rel_err = abs_diff / denom
    mere = np.max(rel_err)
    mare = np.mean(rel_err)

    # Use appropriate tolerance based on dtype
    if dtype_str == 'fp32':
        rtol = 0.001   # 0.1% for fp32
        atol = 0.001
    else:
        rtol = 0.01    # 1% relative tolerance for fp16
        atol = 0.005   # 5e-3 absolute tolerance

    # allclose with bf16 tolerance
    passed = np.allclose(out_f32, gold_f32, rtol=rtol, atol=atol)

    if passed:
        print(f"Verification PASSED! Shape: {output.shape}  dtype: {dtype_str}")
    else:
        print(f"Verification FAILED! Shape: {output.shape}  dtype: {dtype_str}")

    print(f"  Max abs diff: {max_err:.6e}")
    print(f"  Mean abs diff: {mean_err:.6e}")
    print(f"  MERE: {mere:.6e}  (target < 7.81e-3)")
    print(f"  MARE: {mare:.6e}  (target < 7.81e-2)")
    print(f"  allclose (rtol={rtol}, atol={atol}): {'PASS' if passed else 'FAIL'}")

    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin> [dtype=fp16|fp32]")
        sys.exit(1)

    dtype_arg = sys.argv[3] if len(sys.argv) > 3 else 'fp16'
    success = verify_result(sys.argv[1], sys.argv[2], dtype_arg)
    sys.exit(0 if success else 1)
