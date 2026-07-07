# Sinkhorn Normalize - 测试数据生成脚本

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 数据规格: [1, 1024, 4, 4] float32
batch = 1024
mhc = 4
shape = (1, batch, mhc, mhc)
total_elements = np.prod(shape)

# 生成标准正态分布随机输入
np.random.seed(42)
x = np.random.randn(*shape).astype(np.float32)

# 展平后保存为二进制
x_flat = x.flatten()
x_flat.tofile("input/input_x.bin")

# 计算 golden (使用 PyTorch 参考实现)
x_tensor = x  # keep shape for compute_golden
golden = compute_golden(x_tensor, mhc=mhc)
golden_flat = golden.flatten()
golden_flat.tofile("output/golden.bin")

print(f"Generated test data:")
print(f"  shape: {shape}, dtype: float32")
print(f"  total_elements: {total_elements}")
print(f"  input/input_x.bin: {x_flat.nbytes} bytes")
print(f"  output/golden.bin: {golden_flat.nbytes} bytes")
