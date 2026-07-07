# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# SparseAttn PyTorch 通路测试脚本
# ============================================================================

import sys
import os
import numpy as np
import torch
import torch_npu

from golden import compute_golden

# 算子配置
SO_NAME = "libsparse_attn_ops.so"
OP_NAME = "sparse_attn"

MERE_THRESHOLD = 2.0 ** -7    # ≈ 0.00781
MARE_THRESHOLD = 10.0 * MERE_THRESHOLD  # ≈ 0.0781


def run_test(name, q, kv, attn_sink, topk_idxs, softmax_scale):
    """Run a single test case. Returns (name, passed, mere, mare, max_abs)."""
    op_fn = getattr(torch.ops.npu, OP_NAME)

    # Run on NPU
    y = op_fn(q.npu(), kv.npu(), attn_sink.npu(), topk_idxs.npu(), softmax_scale)

    # Compute golden
    q_np = q.float().numpy()
    kv_np = kv.float().numpy()
    sink_np = attn_sink.float().numpy()
    idx_np = topk_idxs.numpy()
    golden = compute_golden(q_np, kv_np, sink_np, idx_np, softmax_scale)

    # Compare
    output_np = y.cpu().float().numpy()
    abs_err = np.abs(output_np - golden)
    denom = np.maximum(np.abs(golden), 1e-8)
    rel_err = abs_err / denom

    mere = np.max(rel_err)
    mare = np.mean(rel_err)
    max_abs = np.max(abs_err)

    passed = (mere <= MERE_THRESHOLD) and (mare <= MARE_THRESHOLD)
    return name, passed, mere, mare, max_abs


def main():
    # Load the custom op library
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"WARNING: {so_path} not found. Build PyTorch extension first.")
        print("Run: cd build && cmake .. && make -j4 sparse_attn_ops")
        sys.exit(0)

    torch.ops.load_library(so_path)

    results = []

    # T1: Default config
    b, m, n, h, d, topk = 2, 16, 32, 8, 64, 16
    softmax_scale = 1.0 / np.sqrt(float(d))
    rng = np.random.RandomState(42)
    q = torch.from_numpy(rng.randn(b, m, h, d).astype(np.float32)).bfloat16()
    kv = torch.from_numpy(rng.randn(b, n, d).astype(np.float32)).bfloat16()
    sink = torch.from_numpy(rng.randn(h).astype(np.float32) * 0.1).float()
    idx = torch.from_numpy(rng.randint(0, n, (b, m, topk)).astype(np.int32)).int()
    mask = rng.random((b, m, topk)) < 0.1
    idx_np = idx.numpy()
    idx_np[mask] = -1
    idx = torch.from_numpy(idx_np).int()
    results.append(run_test("T1 default", q, kv, sink, idx, softmax_scale))

    # T2: Minimal shape
    b, m, n, h, d, topk = 1, 1, 32, 8, 64, 16
    rng = np.random.RandomState(123)
    q = torch.from_numpy(rng.randn(b, m, h, d).astype(np.float32)).bfloat16()
    kv = torch.from_numpy(rng.randn(b, n, d).astype(np.float32)).bfloat16()
    sink = torch.from_numpy(rng.randn(h).astype(np.float32) * 0.1).float()
    idx = torch.from_numpy(np.clip(rng.randint(-2, n, (b, m, topk)), -1, n-1).astype(np.int32)).int()
    results.append(run_test("T2 minimal", q, kv, sink, idx, softmax_scale))

    # Summary
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, mere, mare, max_abs in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} "
              f"(MERE={mere:.6f} MARE={mare:.6f} MaxAbs={max_abs:.6f})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
