# Sinkhorn Normalize - 结果验证脚本

import numpy as np
import sys

# 验证参数: FP32 精度标准
# MERE < 2^-13 ≈ 0.000122, MARE < 10 * 2^-13 ≈ 0.00122
RTOL = 1e-5
ATOL = 1e-6

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    total = len(golden)
    diff = np.abs(output - golden)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # 检查 NaN/Inf
    output_nan = np.sum(np.isnan(output))
    output_inf = np.sum(np.isinf(output))
    golden_nan = np.sum(np.isnan(golden))
    golden_inf = np.sum(np.isinf(golden))

    if output_nan > 0 or output_inf > 0:
        print(f"WARNING: output contains NaN={output_nan}, Inf={output_inf}")
    if golden_nan > 0 or golden_inf > 0:
        print(f"WARNING: golden contains NaN={golden_nan}, Inf={golden_inf}")

    # 相对误差计算 (避免除零)
    denom = np.maximum(np.abs(golden), 1e-10)
    rel_diff = diff / denom
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    # 使用 allclose 进行基本判定
    passed = np.allclose(output, golden, rtol=RTOL, atol=ATOL)

    print(f"Verification:")
    print(f"  Total elements: {total}")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Max relative diff: {max_rel_diff:.6e}")
    print(f"  Mean relative diff: {mean_rel_diff:.6e}")
    print(f"  NaN count (output/golden): {output_nan}/{golden_nan}")
    print(f"  Inf count (output/golden): {output_inf}/{golden_inf}")

    # 精度检查
    mere_ok = mean_rel_diff < 0.000122   # 2^-13
    mare_ok = max_rel_diff < 0.00122     # 10 * 2^-13

    print(f"  MERE check: {'PASS' if mere_ok else 'FAIL'} (mean_rel={mean_rel_diff:.6e} < 0.000122)")
    print(f"  MARE check: {'PASS' if mare_ok else 'FAIL'} (max_rel={max_rel_diff:.6e} < 0.00122)")

    if not passed:
        mismatches = np.where(diff > ATOL + RTOL * np.abs(golden))[0]
        print(f"  Mismatch count: {len(mismatches)} / {total}")
        if len(mismatches) > 0 and len(mismatches) <= 20:
            for idx in mismatches[:10]:
                print(f"    [{idx}] output={output[idx]:.6e} golden={golden[idx]:.6e} diff={diff[idx]:.6e}")

    # 双随机性检查 (仅当 shape 可 reshape 为 [1, batch, 4, 4] 时)
    try:
        n_batch = total // 16
        out_reshaped = output.reshape(1, n_batch, 4, 4)
        golden_reshaped = golden.reshape(1, n_batch, 4, 4)

        out_row_sum = np.sum(out_reshaped, axis=-1)
        out_col_sum = np.sum(out_reshaped, axis=-2)
        out_row_err = np.max(np.abs(out_row_sum - 1.0))
        out_col_err = np.max(np.abs(out_col_sum - 1.0))

        golden_row_sum = np.sum(golden_reshaped, axis=-1)
        golden_col_sum = np.sum(golden_reshaped, axis=-2)
        golden_row_err = np.max(np.abs(golden_row_sum - 1.0))
        golden_col_err = np.max(np.abs(golden_col_sum - 1.0))

        print(f"  Doubly stochastic check (output):")
        print(f"    Max |row_sum - 1|: {out_row_err:.6e}")
        print(f"    Max |col_sum - 1|: {out_col_err:.6e}")
        print(f"  Doubly stochastic check (golden):")
        print(f"    Max |row_sum - 1|: {golden_row_err:.6e}")
        print(f"    Max |col_sum - 1|: {golden_col_err:.6e}")
    except Exception:
        pass

    result = "PASSED" if passed else "FAILED"
    print(f"\nOverall: {result}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
