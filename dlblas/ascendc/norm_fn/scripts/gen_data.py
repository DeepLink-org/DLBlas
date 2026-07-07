# 测试数据生成脚本
# 生成 norm_fn 算子的输入输出数据

import numpy as np
import os
import sys

# Add scripts dir to path for golden import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ============================================================================
# 数据规格参数
# ============================================================================
TOTAL_M = 13
TOTAL_N = 24
TOTAL_K = 5120

# ============================================================================
# 模式选择: 通过命令行参数控制
#   --with-weight   : 生成带有权重的测试数据
#   (默认)           : 生成无权重的测试数据
# ============================================================================
with_weight = "--with-weight" in sys.argv

n0 = 1
mhc_mult = 4  # TOTAL_N = mhc_mult * (2 + mhc_mult) = 4*6 = 24 ✓
hidden_size = 1280  # TOTAL_K = mhc_mult * hidden_size = 4*1280 = 5120 ✓

# 生成 residual (bf16)
residual_f32 = (
    np.random.randn(n0, TOTAL_M, mhc_mult, hidden_size).astype(np.float32)
    * (1 + np.arange(mhc_mult, dtype=np.float32).reshape(1, 1, -1, 1) * 0.01)
)

# 转换为 bfloat16 存储格式
residual_bf16_view = residual_f32.astype(np.float32).view(np.uint32)
residual_bf16_bits = (residual_bf16_view >> 16).astype(np.uint16)
residual_bf16 = residual_bf16_bits.view(np.float16).astype(np.float32)
# 转回 bf16 的 uint16 表示用于存储
residual_bf16_as_uint16 = residual_bf16_bits

# 生成 mhc_fn (float32)
fn_f32 = (
    np.random.randn(TOTAL_N, mhc_mult, TOTAL_K // mhc_mult).astype(np.float32)
    * 1e-4
    * (1 + np.arange(mhc_mult, dtype=np.float32).reshape(1, -1, 1) * 0.01)
).reshape(TOTAL_N, TOTAL_K)  # shape: (24, 5120)

# 生成 weight (可选)
if with_weight:
    weight = (np.random.randn(TOTAL_K).astype(np.float32) * 0.1 + 1.0)
else:
    weight = np.zeros(1, dtype=np.float32)  # placeholder

# 保存输入文件
# residual: bf16 格式，存储为 uint16 (bfloat16)
residual_bf16_as_uint16.tofile("input/input_residual.bin")
fn_f32.tofile("input/input_mhc_fn.bin")

if with_weight:
    weight.tofile("input/input_weight.bin")
else:
    # 无权重时删除 weight 文件并创建占位文件（4 字节），host 通过文件缺失判断
    weight_path = "input/input_weight.bin"
    if os.path.exists(weight_path):
        os.remove(weight_path)
    # 创建占位文件供 host 读取（KernelCall 始终需要 3 个输入）
    np.zeros(1, dtype=np.float32).tofile("input/input_weight_dummy.bin")

# 计算 golden 输出
# 将 bf16 uint16 转回 float32 用于 golden 计算
bf16_as_uint32 = residual_bf16_as_uint16.astype(np.uint32) << 16
residual_f32_from_bf16 = bf16_as_uint32.view(np.float32).reshape(1, TOTAL_M, mhc_mult, hidden_size)

if with_weight:
    golden = compute_golden(residual_bf16_as_uint16, fn_f32, weight, 1e-6)
else:
    golden = compute_golden(residual_bf16_as_uint16, fn_f32, None, 1e-6)

golden.tofile("output/golden.bin")

print(f"Generated test data:")
print(f"  Mode: {'with_weight' if with_weight else 'without_weight'}")
print(f"  residual: shape={residual_bf16_as_uint16.shape}, dtype=bfloat16 (uint16)")
print(f"  mhc_fn: shape={fn_f32.shape}, dtype=float32")
if with_weight:
    print(f"  weight: shape={weight.shape}, dtype=float32")
else:
    print(f"  weight: None (placeholder)")
print(f"  golden: shape={golden.shape}, dtype=float32")
print(f"Files written to input/ and output/")
