#!/usr/bin/env python3
"""Benchmark AscendC kernel vs PyTorch reference for sparse_attn."""

import os
import sys
import time
import json
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(PROJ_DIR, "build")

from golden import compute_golden


def float32_to_bf16(arr_fp32):
    arr_u32 = arr_fp32.astype(np.float32).view(np.uint32)
    return (arr_u32 >> 16).astype(np.uint16)


def bf16_to_float32(arr_bf16):
    arr_u32 = arr_bf16.astype(np.uint32) << 16
    return arr_u32.view(np.float32)


def generate_data(b, m, n, h, d, topk, seed=42):
    """Generate test data matching gen_data.py format."""
    rng = np.random.RandomState(seed)
    softmax_scale = 1.0 / np.sqrt(float(d))

    q = rng.randn(b, m, h, d).astype(np.float32)
    q_bf16 = float32_to_bf16(q)
    kv = rng.randn(b, n, d).astype(np.float32)
    kv_bf16 = float32_to_bf16(kv)
    attn_sink = rng.randn(h).astype(np.float32) * 0.1
    topk_idxs = rng.randint(0, n, (b, m, topk)).astype(np.int32)
    mask = rng.random((b, m, topk)) < 0.1
    topk_idxs[mask] = -1

    # Compute golden
    golden = compute_golden(q, kv, attn_sink, topk_idxs, softmax_scale)
    golden_bf16 = float32_to_bf16(golden)

    # Write binary files for kernel
    os.makedirs(os.path.join(BUILD_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(BUILD_DIR, "output"), exist_ok=True)
    q_bf16.tofile(os.path.join(BUILD_DIR, "input/input_q.bin"))
    kv_bf16.tofile(os.path.join(BUILD_DIR, "input/input_kv.bin"))
    topk_idxs.tofile(os.path.join(BUILD_DIR, "input/input_idx.bin"))
    attn_sink.astype(np.float32).tofile(os.path.join(BUILD_DIR, "input/input_sink.bin"))
    golden_bf16.tofile(os.path.join(BUILD_DIR, "output/golden.bin"))

    return q, kv, attn_sink, topk_idxs, softmax_scale, golden, golden_bf16


def verify_precision(output_uint16, golden_uint16):
    """Verify precision: return mere, mare, max_abs, passed."""
    output_u32 = output_uint16.astype(np.uint32) << 16
    output_fp32 = output_u32.view(np.float32)
    golden_u32 = golden_uint16.astype(np.uint32) << 16
    golden_fp32 = golden_u32.view(np.float32)

    abs_err = np.abs(output_fp32 - golden_fp32)
    denom = np.maximum(np.abs(golden_fp32), 1e-8)
    rel_err = abs_err / denom

    mere = float(np.max(rel_err))
    mare = float(np.mean(rel_err))
    max_abs = float(np.max(abs_err))

    mere_threshold = 2.0 ** -7
    mare_threshold = 10.0 * mere_threshold
    passed = (mere <= mere_threshold) and (mare <= mare_threshold)
    return mere, mare, max_abs, passed


def time_ascend_kernel(exe_path, warmup=5, repeat=20):
    """Time the Ascend C kernel executable. Returns avg time in microseconds."""
    times = []
    for i in range(warmup + repeat):
        # Remove previous output
        output_path = os.path.join(BUILD_DIR, "output/output.bin")
        if os.path.exists(output_path):
            os.remove(output_path)

        start = time.perf_counter()
        result = subprocess.run([exe_path], cwd=BUILD_DIR,
                                capture_output=True, text=True)
        end = time.perf_counter()
        elapsed_us = (end - start) * 1e6

        if result.returncode != 0:
            print(f"  Kernel run {i} failed: {result.stderr[:500]}")
            return None

        if not os.path.exists(output_path):
            print(f"  Kernel run {i} produced no output!")
            return None

        if i >= warmup:
            times.append(elapsed_us)

    if not times:
        return None
    avg_us = sum(times) / len(times)
    return avg_us


def time_pytorch_ref(q, kv, attn_sink, topk_idxs, softmax_scale, warmup=20, repeat=100):
    """Time PyTorch reference implementation. Returns avg time in microseconds."""
    import torch

    q_t = torch.from_numpy(q).bfloat16()
    kv_t = torch.from_numpy(kv).bfloat16()
    sink_t = torch.from_numpy(attn_sink).float()
    idx_t = torch.from_numpy(topk_idxs).int()

    def sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, m, h, d = q.shape
        topk = topk_idxs.shape[-1]
        valid_mask = topk_idxs >= 0
        safe_idxs = topk_idxs.clamp(min=0).long()
        b_idx = torch.arange(b, device=q.device)[:, None, None].expand(b, m, topk)
        gathered_kv = kv[b_idx, safe_idxs]
        gathered_kv = gathered_kv.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
        scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))
        sink = attn_sink.float().view(1, 1, h, 1)
        max_scores = torch.amax(scores, dim=-1, keepdim=True)
        max_scores = torch.maximum(max_scores, sink)
        exp_scores = torch.exp(scores - max_scores)
        exp_scores = exp_scores.masked_fill(~valid_mask.unsqueeze(2), 0.0)
        exp_sink = torch.exp(sink - max_scores)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
        attn_weights = exp_scores / sum_exp
        output = torch.einsum("bmht,bmtd->bmhd", attn_weights, gathered_kv.float())
        return output.to(q.dtype)

    # Warmup
    for _ in range(warmup):
        _ = sparse_attn_ref(q_t, kv_t, sink_t, idx_t, softmax_scale)

    # Measure
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = sparse_attn_ref(q_t, kv_t, sink_t, idx_t, softmax_scale)
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    avg_us = sum(times) / len(times)
    return avg_us


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--h", type=int, default=8)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--skip-kernel", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    b, m, n, h, d, topk = args.b, args.m, args.n, args.h, args.d, args.topk
    print(f"Benchmark: b={b} m={m} n={n} h={h} d={d} topk={topk}")

    # Generate data
    print("Generating test data...")
    q, kv, attn_sink, topk_idxs, softmax_scale, golden, golden_bf16 = \
        generate_data(b, m, n, h, d, topk)
    total_elements = b * m * h * d
    print(f"  Output shape: [{b}, {m}, {h}, {d}] ({total_elements} elements)")

    # Time PyTorch reference
    print(f"\nTiming PyTorch reference ({args.warmup} warmup + {args.repeat} runs)...")
    torch_us = time_pytorch_ref(q, kv, attn_sink, topk_idxs, softmax_scale,
                                warmup=args.warmup, repeat=args.repeat)
    print(f"  PyTorch avg: {torch_us:.2f} us")

    # Time AscendC kernel
    ascend_us = None
    precision_ok = False
    mere = mare = max_abs = None

    if not args.skip_kernel:
        exe_path = os.path.join(BUILD_DIR, "sparse_attn_custom")
        if not os.path.exists(exe_path):
            print(f"\nWARNING: {exe_path} not found, skipping kernel benchmark")
        else:
            print(f"\nTiming AscendC kernel ({args.warmup} warmup + {args.repeat} runs)...")
            ascend_us = time_ascend_kernel(exe_path, warmup=args.warmup, repeat=args.repeat)
            if ascend_us is not None:
                print(f"  AscendC avg: {ascend_us:.2f} us")

                # Verify precision
                output_path = os.path.join(BUILD_DIR, "output/output.bin")
                output_uint16 = np.fromfile(output_path, dtype=np.uint16)
                golden_path = os.path.join(BUILD_DIR, "output/golden.bin")
                golden_uint16_r = np.fromfile(golden_path, dtype=np.uint16)
                mere, mare, max_abs, precision_ok = verify_precision(output_uint16, golden_uint16_r)
                print(f"  Precision: MERE={mere:.6f} MARE={mare:.6f} MaxAbs={max_abs:.6f} "
                      f"{'PASS' if precision_ok else 'FAIL'}")
            else:
                print("  AscendC kernel FAILED")

    # Summary
    speedup = None
    if ascend_us is not None and torch_us is not None and torch_us > 0:
        speedup = torch_us / ascend_us

    print(f"\n{'='*60}")
    print(f"Benchmark Summary")
    print(f"{'='*60}")
    print(f"  PyTorch CPU: {torch_us:.2f} us")
    if ascend_us is not None:
        print(f"  AscendC NPU: {ascend_us:.2f} us")
        print(f"  Speedup:     {speedup:.4f}x" if speedup else "  Speedup: N/A")
    print(f"  Precision:   MERE={mere} MARE={mare} MaxAbs={max_abs}")

    # Output JSON
    result = {
        "ascend_us": ascend_us,
        "torch_us": torch_us,
        "speedup_vs_torch": speedup,
        "precision": {
            "mere": mere,
            "mare": mare,
            "max_abs": max_abs,
            "passed": precision_ok,
        },
        "shape": {"b": b, "m": m, "n": n, "h": h, "d": d, "topk": topk},
    }

    output_file = args.output or os.path.join(PROJ_DIR, "benchmark_result.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {output_file}")

    return result


if __name__ == "__main__":
    main()
