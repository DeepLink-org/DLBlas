# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 测试数据生成 - Expand Kernel Backward 算子
# 输入: o_grad shape (n0, n1, mhc_mult, h)
# 输出: sum(o_grad, dim=-2)
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Shape 参数: (n0, n1, mhc_mult, h) = (2, 1024, 4, 1280)
n0 = 2
n1 = 1024
mhc_mult = 4
h = 1280

dtype = np.float16

# 生成输入数据 o_grad
o_grad = np.random.randn(n0, n1, mhc_mult, h).astype(dtype)

o_grad.tofile("input/input_o_grad.bin")

# 计算 golden: sum along dim=-2
golden = compute_golden(o_grad)
golden.tofile("output/golden.bin")

print(f"Generated test data:")
print(f"  Input shape:  {o_grad.shape} ({n0}, {n1}, {mhc_mult}, {h})")
print(f"  Output shape: {golden.shape} ({n0}, {n1}, {h})")
print(f"  Dtype: {dtype}")
print(f"  input/input_o_grad.bin: {o_grad.shape}, {o_grad.dtype}")
print(f"  output/golden.bin:     {golden.shape}, {golden.dtype}")
