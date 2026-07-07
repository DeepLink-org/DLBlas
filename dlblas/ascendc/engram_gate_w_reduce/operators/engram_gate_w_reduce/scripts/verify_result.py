# ----------------------------------------------------------------------------------------------------------
# engram_gate_w_reduce 结果验证脚本
#
# 验证 kernel 输出的 grad_weight_hidden, grad_weight_embed 与 golden 一致
# ----------------------------------------------------------------------------------------------------------

import numpy as np
import sys
import os

# 验证参数
DTYPE = np.float32
RTOL = 1e-4     # 相对容差 (考虑 BF16→FP32 转换的固有误差)
ATOL = 1e-6     # 绝对容差


def verify_output(output_path, golden_path, name):
    """验证单个输出"""
    output = np.fromfile(output_path, dtype=DTYPE)
    golden = np.fromfile(golden_path, dtype=DTYPE)

    if output.shape != golden.shape:
        print(f"[{name}] Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    diff = np.abs(output - golden)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    if np.allclose(output, golden, rtol=RTOL, atol=ATOL):
        print(f"[{name}] PASSED! Shape={output.shape}, max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        return True
    else:
        print(f"[{name}] FAILED! Shape={output.shape}")
        print(f"  max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        mismatches = np.where(diff > ATOL + RTOL * np.abs(golden))[0]
        print(f"  mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0 and len(mismatches) <= 10:
            for idx in mismatches[:10]:
                print(f"    [{idx}] output={output[idx]:.6f}, golden={golden[idx]:.6f}, diff={diff[idx]:.6e}")
        return False


def main():
    output_dir = "output"
    results = []

    # 验证 grad_weight_hidden
    r1 = verify_output(
        os.path.join(output_dir, "output_grad_weight_hidden.bin"),
        os.path.join(output_dir, "golden_grad_weight_hidden.bin"),
        "grad_weight_hidden")
    results.append(r1)

    # 验证 grad_weight_embed
    r2 = verify_output(
        os.path.join(output_dir, "output_grad_weight_embed.bin"),
        os.path.join(output_dir, "golden_grad_weight_embed.bin"),
        "grad_weight_embed")
    results.append(r2)

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Verification: {passed}/{total} passed")
    print(f"Status: {'PASSED' if passed == total else 'FAILED'}")
    print(f"{'='*50}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
