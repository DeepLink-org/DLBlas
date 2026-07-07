# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# SparseAttn 测试数据生成脚本
# ============================================================================

import numpy as np
import os
import sys

# Add parent directory for golden module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden


def float32_to_bf16(arr_fp32):
    """Convert float32 numpy array to bfloat16 (uint16) representation.

    bf16 uses the upper 16 bits of fp32. This is NOT the same as fp16.
    """
    arr_u32 = arr_fp32.astype(np.float32).view(np.uint32)
    arr_bf16 = (arr_u32 >> 16).astype(np.uint16)
    return arr_bf16


def bf16_to_float32(arr_bf16):
    """Convert bfloat16 (uint16) representation back to float32."""
    arr_u32 = arr_bf16.astype(np.uint32) << 16
    arr_fp32 = arr_u32.view(np.float32)
    return arr_fp32

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def gen_test_data(b=2, m=16, n=32, h=8, d=64, topk=16, seed=42):
    """Generate test data for sparse_attn.

    Args:
        b, m, n, h, d, topk: shape parameters
        seed: random seed for reproducibility

    Returns:
        dict with all generated data and golden output
    """
    rng = np.random.RandomState(seed)

    softmax_scale = 1.0 / np.sqrt(float(d))

    # Q: [b, m, h, d] bf16
    q = rng.randn(b, m, h, d).astype(np.float32)
    q_bf16 = float32_to_bf16(q)

    # KV: [b, n, d] bf16
    kv = rng.randn(b, n, d).astype(np.float32)
    kv_bf16 = float32_to_bf16(kv)

    # attn_sink: [h] fp32
    attn_sink = rng.randn(h).astype(np.float32) * 0.1

    # topk_idxs: [b, m, topk] int32, some -1 for invalid
    topk_idxs = rng.randint(0, n, (b, m, topk)).astype(np.int32)
    # Randomly set ~10% to -1
    mask = rng.random((b, m, topk)) < 0.1
    topk_idxs[mask] = -1

    # Compute golden using reference implementation
    golden = compute_golden(q, kv, attn_sink, topk_idxs, softmax_scale)
    # Golden is float32, convert to bf16 (uint16 representation)
    golden_bf16 = float32_to_bf16(golden)

    data = {
        'q': q_bf16,
        'kv': kv_bf16,
        'topk_idxs': topk_idxs,
        'attn_sink': attn_sink.astype(np.float32),
        'golden_bf16': golden_bf16,
        'golden_fp32': golden,
        'softmax_scale': softmax_scale,
    }
    return data


if __name__ == "__main__":
    # Default config
    b, m, n, h, d, topk = 2, 16, 32, 8, 64, 16

    if len(sys.argv) >= 7:
        b    = int(sys.argv[1])
        m    = int(sys.argv[2])
        n    = int(sys.argv[3])
        h    = int(sys.argv[4])
        d    = int(sys.argv[5])
        topk = int(sys.argv[6])

    print(f"Generating test data: b={b} m={m} n={n} h={h} d={d} topk={topk}")

    data = gen_test_data(b, m, n, h, d, topk)

    # Write input files
    data['q'].tofile("input/input_q.bin")
    print(f"  input/input_q.bin:     {data['q'].shape}, {data['q'].dtype}")

    data['kv'].tofile("input/input_kv.bin")
    print(f"  input/input_kv.bin:    {data['kv'].shape}, {data['kv'].dtype}")

    data['topk_idxs'].tofile("input/input_idx.bin")
    print(f"  input/input_idx.bin:   {data['topk_idxs'].shape}, {data['topk_idxs'].dtype}")

    data['attn_sink'].tofile("input/input_sink.bin")
    print(f"  input/input_sink.bin:  {data['attn_sink'].shape}, {data['attn_sink'].dtype}")

    # Write golden output (bf16 representation)
    data['golden_bf16'].tofile("output/golden.bin")
    print(f"  output/golden.bin:     {data['golden_bf16'].shape}, {data['golden_bf16'].dtype}")
    print("Done.")
