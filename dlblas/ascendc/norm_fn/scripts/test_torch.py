# PyTorch 通路测试脚本
# 测试 torch.ops.npu.norm_fn 算子

import numpy as np
import torch
import torch_npu
import os
import sys

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from golden import compute_golden_torch

# 加载自定义算子
torch.ops.load_library("libnorm_fn_ops.so")

# 参数
TOTAL_M = 13
TOTAL_N = 24
TOTAL_K = 5120
mhc_mult = 4  # TOTAL_N = mhc_mult * (2 + mhc_mult)

with_weight = "--with-weight" in sys.argv
eps = 1e-6

print(f"Testing norm_fn (PyTorch pathway), with_weight={with_weight}")

# 生成测试数据
n0 = 1

# residual: (1, 13, 4, 1280) bfloat16
residual_f32 = torch.randn(n0, TOTAL_M, mhc_mult, TOTAL_K // mhc_mult, dtype=torch.float32)
residual_f32 = residual_f32 * (1 + torch.arange(mhc_mult, dtype=torch.float32).view(1, 1, -1, 1) * 0.01)
residual_bf16 = residual_f32.bfloat16().npu()

# mhc_fn: (24, 5120) float32
mhc_fn_f32 = torch.randn(TOTAL_N, TOTAL_K, dtype=torch.float32) * 1e-4
mhc_fn_npu = mhc_fn_f32.npu()

# weight (optional)
if with_weight:
    weight_f32 = torch.randn(TOTAL_K, dtype=torch.float32) * 0.1 + 1.0
    weight_npu = weight_f32.npu()
else:
    weight_npu = None

# 调用算子
result = torch.ops.npu.norm_fn(residual_bf16, mhc_fn_npu, weight_npu, eps)

print(f"Output shape: {result.shape}, dtype: {result.dtype}")
assert result.is_npu, "Output not on NPU!"

# Golden 参考计算 (CPU) - 使用 float32 输入版本
residual_np = residual_bf16.cpu().float().numpy()
mhc_fn_np = mhc_fn_npu.cpu().numpy()
if with_weight:
    weight_np = weight_f32.numpy()
    golden_np = compute_golden_torch(residual_np, mhc_fn_np, weight_np, eps)
else:
    golden_np = compute_golden_torch(residual_np, mhc_fn_np, None, eps)

# 对比
result_cpu = result.cpu().numpy()

max_diff = np.max(np.abs(result_cpu - golden_np))
mean_diff = np.mean(np.abs(result_cpu - golden_np))

print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

rtol = 1e-3
atol = 1e-4  # 与 DESIGN.md Max Diff < 1e-4 一致

if np.allclose(result_cpu, golden_np, rtol=rtol, atol=atol):
    print("PyTorch pathway verification PASSED!")
else:
    mismatches = np.where(np.abs(result_cpu - golden_np) > atol + rtol * np.maximum(np.abs(golden_np), 1e-8))[0]
    print(f"PyTorch pathway verification FAILED! Mismatch count: {len(mismatches)} / {golden_np.size}")
    for i in mismatches[:10]:
        print(f"  [{i}]: result={result_cpu.flat[i]:.8f}, golden={golden_np.flat[i]:.8f}")
    sys.exit(1)
