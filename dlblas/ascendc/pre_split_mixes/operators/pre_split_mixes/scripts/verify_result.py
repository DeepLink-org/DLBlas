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
# 结果验证脚本 — pre_split_mixes (三输出)
# 用法: python3 verify_result.py <output_dir> <golden_dir>
# ============================================================================

import numpy as np
import sys
import os

DTYPE = np.float32
RTOL = 1e-4   # FP32 社区标准
ATOL = 1e-6


def verify_one(name, output_path, golden_path):
    if not os.path.exists(output_path):
        print(f"  {name}: SKIPPED (output file not found: {output_path})")
        return False, 0.0
    output = np.fromfile(output_path, dtype=DTYPE)
    golden = np.fromfile(golden_path, dtype=DTYPE)

    if output.shape != golden.shape:
        print(f"  {name}: FAILED - Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False, float('nan')

    max_diff = np.max(np.abs(output - golden))
    passed = np.allclose(output, golden, rtol=RTOL, atol=ATOL)

    status = "PASSED" if passed else "FAILED"
    print(f"  {name}: {status} (max_diff={max_diff:.6e}, shape={output.shape})")

    if not passed:
        diff = np.abs(output - golden)
        mismatches = np.where(diff > ATOL + RTOL * np.abs(golden))
        print(f"    Mismatch count: {len(mismatches[0])} / {len(golden)}")
        if len(mismatches[0]) > 0:
            print(f"    First mismatch indices: {mismatches[0][:5]}")
            print(f"    Output values: {output[mismatches[0][:5]]}")
            print(f"    Golden values: {golden[mismatches[0][:5]]}")

    return passed, max_diff


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    golden_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    print("=" * 60)
    print("pre_split_mixes Verification")
    print(f"rtol={RTOL}, atol={ATOL}")
    print("=" * 60)

    results = []
    for name, out_fname, gold_fname in [
        ("pre_mix",  "pre_mix.bin",  "pre_mix.bin"),
        ("post_mix", "post_mix.bin", "post_mix.bin"),
        ("comb_mix", "comb_mix.bin", "comb_mix.bin"),
    ]:
        ok, diff = verify_one(
            name,
            os.path.join(output_dir, out_fname),
            os.path.join(golden_dir, gold_fname),
        )
        results.append((name, ok, diff))

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print("=" * 60)
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
