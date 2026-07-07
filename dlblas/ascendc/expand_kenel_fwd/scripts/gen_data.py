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
# 测试数据生成脚本 - Expand 算子
# ============================================================================
#
# 用法: python3 gen_data.py <B> <S> <H> <M> [dtype]
#   dtype: fp16 (默认) 或 fp32
# ============================================================================

import numpy as np
import sys
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 解析命令行参数
if len(sys.argv) < 5:
    print("Usage: python3 gen_data.py <B> <S> <H> <M> [dtype]")
    sys.exit(1)

B = int(sys.argv[1])
S = int(sys.argv[2])
H = int(sys.argv[3])
M = int(sys.argv[4])
dtype_str = sys.argv[5] if len(sys.argv) >= 6 else "fp16"

if dtype_str == "fp32":
    dtype = np.float32
else:
    dtype = np.float16

# 生成随机输入数据，形状 (B, S, H)
x = np.random.randn(B, S, H).astype(dtype)
x.tofile("input/x_input.bin")

# 计算 golden 输出: unsqueeze(-2).expand(...)
golden = compute_golden(x, M)
golden.tofile("output/golden.bin")

print(f"Generated test data: B={B} S={S} H={H} M={M} dtype={dtype_str}")
print(f"  input/x_input.bin: shape={x.shape}, dtype={x.dtype}, size={x.size}")
print(f"  output/golden.bin: shape={golden.shape}, dtype={golden.dtype}, size={golden.size}")
