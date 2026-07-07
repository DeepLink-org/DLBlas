# 结果验证脚本
# 对比算子输出与 golden 参考输出

import numpy as np
import sys

dtype = np.float32
rtol = 1e-3  # 相对容差 (bf16 输入 → float32 计算)
atol = 1e-4  # 绝对容差 (与 DESIGN.md Max Diff < 1e-4 一致)


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {np.max(np.abs(output - golden)):.6e}")
        return True
    else:
        diff = np.abs(output - golden)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"Verification FAILED!")
        print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

        # 找出差异最大的元素
        mismatches = np.where(diff > atol + rtol * np.maximum(np.abs(golden), 1e-8))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")

        if len(mismatches) > 0:
            print(f"First 10 mismatches:")
            for i in mismatches[:10]:
                print(f"  [{i}]: output={output[i]:.8f}, golden={golden[i]:.8f}, diff={diff[i]:.8f}")

        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
