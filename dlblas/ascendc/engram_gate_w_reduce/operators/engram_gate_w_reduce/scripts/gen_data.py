# ----------------------------------------------------------------------------------------------------------
# engram_gate_w_reduce 测试数据生成
#
# 输入:
#   grad_w_partial:   [108, 4, hidden_size] float32
#   weight_hidden:    [4, hidden_size] bfloat16
#   weight_embed:     [4, hidden_size] bfloat16
#   grad_weight_hidden: [4, hidden_size] float32 (in-place)
#   grad_weight_embed:  [4, hidden_size] float32 (in-place)
#
# 输出:
#   grad_weight_hidden: [4, hidden_size] float32
#   grad_weight_embed:  [4, hidden_size] float32
# ----------------------------------------------------------------------------------------------------------

import numpy as np
import os
import argparse

from golden import compute_golden

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=4096)
args = parser.parse_args()

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

R = 108
C = 4
H = args.hidden_size

# 生成随机输入数据
grad_w_partial = np.random.randn(R, C, H).astype(np.float32)
weight_hidden = np.random.randn(C, H).astype(np.float32)
weight_embed = np.random.randn(C, H).astype(np.float32)
grad_weight_hidden_in = np.random.randn(C, H).astype(np.float32)
grad_weight_embed_in = np.random.randn(C, H).astype(np.float32)

# 保存 FP32 输入
grad_w_partial.tofile("input/input_grad_w_partial.bin")
grad_weight_hidden_in.tofile("input/input_grad_weight_hidden.bin")
grad_weight_embed_in.tofile("input/input_grad_weight_embed.bin")

# weight 保存为 bfloat16
# bfloat16 是截断的 float32：保留高 16 位
def to_bfloat16(arr_fp32):
    """将 float32 数组转换为 bfloat16 (uint16_t 表示)"""
    arr_u32 = arr_fp32.view(np.uint32)
    arr_bf16 = (arr_u32 >> 16).astype(np.uint16)
    return arr_bf16

weight_hidden_bf16 = to_bfloat16(weight_hidden)
weight_embed_bf16 = to_bfloat16(weight_embed)

weight_hidden_bf16.tofile("input/input_weight_hidden.bin")
weight_embed_bf16.tofile("input/input_weight_embed.bin")

# 计算 golden (FP32 全精度)
grad_weight_hidden_out, grad_weight_embed_out = compute_golden(
    grad_w_partial, weight_hidden, weight_embed,
    grad_weight_hidden_in, grad_weight_embed_in)

grad_weight_hidden_out.tofile("output/golden_grad_weight_hidden.bin")
grad_weight_embed_out.tofile("output/golden_grad_weight_embed.bin")

print(f"Generated test data: hidden_size={H}, R={R}, C={C}")
print(f"  grad_w_partial: {grad_w_partial.shape}, {grad_w_partial.dtype}")
print(f"  weight_hidden: {weight_hidden.shape}, bfloat16")
print(f"  weight_embed: {weight_embed.shape}, bfloat16")
print(f"  grad_weight_hidden (in): {grad_weight_hidden_in.shape}, {grad_weight_hidden_in.dtype}")
print(f"  grad_weight_embed (in): {grad_weight_embed_in.shape}, {grad_weight_embed_in.dtype}")
print(f"  golden_grad_weight_hidden: {grad_weight_hidden_out.shape}")
print(f"  golden_grad_weight_embed: {grad_weight_embed_out.shape}")
