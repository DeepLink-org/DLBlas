# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 精度验证 - Expand Kernel Backward 算子
# ============================================================================

import numpy as np
import sys

dtype = np.float16
rtol = 1e-3   # FP16 相对容差
atol = 1e-4   # FP16 绝对容差

# Shape 参数
n0, n1, mhc_mult, h = 2, 1024, 4, 1280
output_shape = (n0, n1, h)


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype).reshape(output_shape)
    golden = np.fromfile(golden_path, dtype=dtype).reshape(output_shape)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    if np.allclose(output.astype(np.float32), golden.astype(np.float32),
                   rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        return True
    else:
        print(f"Verification FAILED!")
        print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden.astype(np.float32)))[0]
        print(f"Mismatch count: {len(mismatches)} / {golden.size}")
        if len(mismatches) > 0 and len(mismatches) <= 10:
            for i in mismatches[:10]:
                idx = np.unravel_index(i, output_shape)
                print(f"  [{idx}]: output={output[idx]:.6f}, golden={golden[idx]:.6f}, diff={diff.flatten()[i]:.6e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
