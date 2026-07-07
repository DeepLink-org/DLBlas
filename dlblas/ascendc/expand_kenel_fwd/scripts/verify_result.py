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
# 结果验证脚本 - Expand 算子（Bitwise Match）
# ============================================================================
#
# 用法: python3 verify_result.py <output.bin> <golden.bin> <dtype>
#   dtype: fp16 或 fp32
# ============================================================================

import numpy as np
import sys


def verify_result(output_path, golden_path, dtype):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    # Bitwise match (非计算类算子标准)
    is_pass = np.array_equal(output, golden)
    mismatches = np.sum(output != golden)

    if is_pass:
        print(f"Verification PASSED! (bitwise match)")
        print(f"  Shape: {output.shape}")
        print(f"  Total elements: {output.size}")
        return True
    else:
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        print(f"Verification FAILED!")
        print(f"  Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
        print(f"  Mismatch count: {mismatches} / {output.size}")
        # 显示前几个差异位置
        mismatch_indices = np.where(output != golden)[0]
        if len(mismatch_indices) > 0:
            first_n = min(5, len(mismatch_indices))
            print(f"  First {first_n} mismatch indices: {mismatch_indices[:first_n]}")
            for idx in mismatch_indices[:first_n]:
                print(f"    [{idx}] output={output[idx]} golden={golden[idx]}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <output.bin> <golden.bin> [dtype]")
        sys.exit(1)

    output_path = sys.argv[1]
    golden_path = sys.argv[2]
    dtype_str = sys.argv[3] if len(sys.argv) >= 4 else "fp16"
    dtype = np.float32 if dtype_str == "fp32" else np.float16

    success = verify_result(output_path, golden_path, dtype)
    sys.exit(0 if success else 1)
