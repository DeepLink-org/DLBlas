# ----------------------------------------------------------------------------
# 测试数据生成 - act_quant_kernel (Single config)
# ----------------------------------------------------------------------------

import numpy as np
import os
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def bf16_encode(arr_f32):
    u32 = arr_f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16).tobytes()


def bf16_decode(arr_bf16):
    return (arr_bf16.astype(np.uint32) << 16).view(np.float32)


def generate_test(total_elements, group_size, eps=1e-10, use_zeros=False,
                  use_extreme=False, use_nan=False):
    """Generate test data and golden for one configuration."""
    if use_zeros:
        x_f32 = np.zeros(total_elements, dtype=np.float32)
    elif use_extreme:
        x_f32 = np.random.randn(total_elements).astype(np.float32) * 3.0
        x_f32[0] = 400.0
        x_f32[1] = -400.0
        x_f32[2] = 0.0001
        x_f32[3] = -0.0001
    else:
        np.random.seed(42)
        x_f32 = np.random.randn(total_elements).astype(np.float32) * 3.0

    if use_nan:
        x_f32[0] = np.nan

    x_bytes = bf16_encode(x_f32)
    bf16_arr = np.frombuffer(x_bytes, dtype=np.uint16)
    x_recovered = bf16_decode(bf16_arr)

    q_golden, s_golden = compute_golden(x_recovered, group_size, eps, False)

    with open("input/input_x.bin", "wb") as f:
        f.write(x_bytes)
    q_golden.tofile("output/golden_q.bin")
    s_golden.tofile("output/golden_s.bin")

    print(f"Generated: {total_elements} elements, group_size={group_size}, "
          f"num_groups={s_golden.size}, {'zeros' if use_zeros else 'random'}")

    return x_bytes, q_golden, s_golden


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    gs = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    mode = sys.argv[3] if len(sys.argv) > 3 else "random"

    generate_test(total, gs,
                  use_zeros=(mode == "zeros"),
                  use_extreme=(mode == "extreme"))
