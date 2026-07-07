# apply_mix test data generation (bf16 x, fp32 mix, bf16 golden)

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden, fp32_to_bf16_uint16

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def generate_test_data(n0, n1, mhc, h):
    # x: sigmoid of random normal, stored as bf16 uint16
    x_fp32 = np.random.randn(n0, n1, mhc, h).astype(np.float32)
    x_fp32 = 1.0 / (1.0 + np.exp(-x_fp32))
    x_bf16 = fp32_to_bf16_uint16(x_fp32).reshape(n0, n1, mhc, h)

    # mix: softmax over mhc axis, stored as fp32
    mix_fp32_raw = np.random.randn(n0, n1, mhc, 1).astype(np.float32)
    mix_max = np.max(mix_fp32_raw, axis=2, keepdims=True)
    mix_exp = np.exp(mix_fp32_raw - mix_max)
    mix_sum = np.sum(mix_exp, axis=2, keepdims=True)
    mix_fp32 = (mix_exp / mix_sum).astype(np.float32)

    # Golden: compute in fp32, then truncate to bf16
    golden = compute_golden(x_bf16, mix_fp32, n0, n1, mhc, h)

    return x_bf16, mix_fp32, golden


n0, n1, mhc, h = 2, 1024, 4, 1280
print(f"Generating: n0={n0}, n1={n1}, mhc={mhc}, h={h}")

x_bf16, mix_fp32, golden = generate_test_data(n0, n1, mhc, h)

shape = np.array([n0, n1, mhc, h], dtype=np.uint32)
shape.tofile("input/shape.bin")
x_bf16.tofile("input/input_x.bin")
mix_fp32.tofile("input/input_mix.bin")
golden.tofile("output/golden.bin")

print(f"  input/input_x.bin: {x_bf16.shape}, uint16 (bf16), {x_bf16.nbytes} bytes")
print(f"  input/input_mix.bin: {mix_fp32.shape}, float32, {mix_fp32.nbytes} bytes")
print(f"  output/golden.bin: {golden.shape}, uint16 (bf16), {golden.nbytes} bytes")
print("Done.")
