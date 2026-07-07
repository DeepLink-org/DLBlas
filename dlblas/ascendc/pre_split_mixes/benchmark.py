#!/usr/bin/env python3
"""Benchmark AscendC pre_split_mixes kernel vs PyTorch reference.
Uses the benchmark binary (pre_split_mixes_bench) which measures kernel-only time
with internal warmup + repeat loops, eliminating process-spawn overhead."""

import subprocess
import time
import os
import sys
import json
import re
import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from golden import compute_golden

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")

TEST_CASES = {
    "T1": {"batch": 1, "seq_len": 1,    "m": 4},
    "T2": {"batch": 1, "seq_len": 1024, "m": 4},
    "T3": {"batch": 8, "seq_len": 512,  "m": 4},
    "T4": {"batch": 1, "seq_len": 2048, "m": 4},
    "T5": {"batch": 1, "seq_len": 1024, "m": 1},
    "T6": {"batch": 1, "seq_len": 1024, "m": 8},
    "T7": {"batch": 1, "seq_len": 1024, "m": 16},
    "T8": {"batch": 2, "seq_len": 256,  "m": 4},
}

DEFAULT_EPS = 1e-2
DEFAULT_POST_MULT = 2.0
DTYPE = torch.float32


def run_ascendc_bench(case_id, tc):
    """Run AscendC benchmark binary and parse avg_us from BENCH_RESULT line."""
    # Generate data (reuse gen_data.py)
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "gen_data.py"), case_id],
        cwd=BUILD_DIR, capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"gen_data failed: {r.stderr}")

    bench_bin = os.path.join(BUILD_DIR, "pre_split_mixes_bench")
    if not os.path.exists(bench_bin):
        raise FileNotFoundError(f"Benchmark binary not found: {bench_bin}")

    r = subprocess.run([bench_bin], cwd=BUILD_DIR, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"Benchmark binary failed (exit {r.returncode}): {r.stderr}")

    # Parse BENCH_RESULT: avg_us=XX.XX total_us=XX.XX iters=N blockNum=N
    m = re.search(r"BENCH_RESULT:\s+avg_us=([\d.]+)\s+total_us=([\d.]+)\s+iters=(\d+)\s+blockNum=(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"Failed to parse BENCH_RESULT from: {r.stdout[-500:]}")

    avg_us = float(m.group(1))
    total_us = float(m.group(2))
    iters = int(m.group(3))
    block_num = int(m.group(4))
    return avg_us, total_us, iters, block_num


def run_torch_bench(tc):
    """Run PyTorch reference and measure time with synchronize."""
    batch, seq_len, m_val = tc["batch"], tc["seq_len"], tc["m"]
    M3 = 2 * m_val + m_val * m_val

    np.random.seed(42)
    input_mixes = torch.from_numpy(
        np.random.randn(batch, seq_len, M3).astype(np.float32)).npu()
    mhc_scale = torch.from_numpy(
        (np.random.randn(3) * 0.1).astype(np.float32)).npu()
    mhc_base = torch.from_numpy(
        (np.random.randn(M3) * 0.1).astype(np.float32)).npu()

    scale_cat = torch.cat([
        mhc_scale[0].expand(m_val),
        mhc_scale[1].expand(m_val),
        mhc_scale[2].expand(m_val * m_val),
    ])

    WARMUP = 20
    REPEAT = 200

    # Warmup
    for _ in range(WARMUP):
        x = input_mixes * scale_cat + mhc_base
        pre = x[:, :, :m_val].sigmoid().unsqueeze(-1) + DEFAULT_EPS
        post = (x[:, :, m_val:2*m_val].sigmoid() * DEFAULT_POST_MULT).unsqueeze(-1)
        comb = x[:, :, 2*m_val:].reshape(batch, seq_len, m_val, m_val)
        torch.npu.synchronize()

    # Measure (use CUDA event-style timing for accuracy)
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(REPEAT)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(REPEAT)]

    for i in range(REPEAT):
        start_events[i].record()
        x = input_mixes * scale_cat + mhc_base
        pre = x[:, :, :m_val].sigmoid().unsqueeze(-1) + DEFAULT_EPS
        post = (x[:, :, m_val:2*m_val].sigmoid() * DEFAULT_POST_MULT).unsqueeze(-1)
        comb = x[:, :, 2*m_val:].reshape(batch, seq_len, m_val, m_val)
        end_events[i].record()

    torch.npu.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    avg_us = float(np.mean(times_us))
    min_us = float(np.min(times_us))
    return avg_us, min_us


def main():
    print("=" * 70)
    print("pre_split_mixes Benchmark: AscendC vs PyTorch")
    print("=" * 70)

    results = []
    for case_id in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
        tc = TEST_CASES[case_id]
        batch, seq_len, m_val = tc["batch"], tc["seq_len"], tc["m"]
        M3 = 2 * m_val + m_val * m_val
        total_rows = batch * seq_len
        total_elems = total_rows * M3

        print(f"\n--- {case_id}: batch={batch}, seq_len={seq_len}, m={m_val} "
              f"(M3={M3}, rows={total_rows}, elems={total_elems}) ---")

        # AscendC kernel benchmark
        asc_data = None
        try:
            avg_us, total_us, iters, block_num = run_ascendc_bench(case_id, tc)
            asc_data = {"avg_us": avg_us, "iters": iters, "blocks": block_num}
            print(f"  AscendC:  avg={avg_us:.2f} us  (iters={iters}, blocks={block_num})")
        except Exception as e:
            print(f"  AscendC:  FAILED - {e}")

        # PyTorch reference benchmark
        torch_data = None
        try:
            avg_us, min_us = run_torch_bench(tc)
            torch_data = {"avg_us": avg_us, "min_us": min_us}
            print(f"  PyTorch:  avg={avg_us:.2f} us  (min={min_us:.2f} us)")
        except Exception as e:
            print(f"  PyTorch:  FAILED - {e}")

        speedup = None
        if asc_data and torch_data and asc_data["avg_us"] > 0:
            speedup = torch_data["avg_us"] / asc_data["avg_us"]
            print(f"  Speedup (AscendC vs PyTorch): {speedup:.4f}x")

        results.append({
            "case_id": case_id,
            "batch": batch, "seq_len": seq_len, "m": m_val,
            "M3": M3, "total_rows": total_rows,
            "ascendc_avg_us": round(asc_data["avg_us"], 2) if asc_data else None,
            "ascendc_blocks": asc_data["blocks"] if asc_data else None,
            "torch_avg_us": round(torch_data["avg_us"], 2) if torch_data else None,
            "torch_min_us": round(torch_data["min_us"], 2) if torch_data else None,
            "speedup_vs_torch": round(speedup, 4) if speedup else None,
        })

    # Compute aggregate metrics
    valid = [r for r in results if r["speedup_vs_torch"] is not None and r["speedup_vs_torch"] > 0]
    if valid:
        geo_mean = float(np.exp(np.mean(np.log([r["speedup_vs_torch"] for r in valid]))))
        total_asc = sum(r["ascendc_avg_us"] for r in valid)
        total_torch = sum(r["torch_avg_us"] for r in valid)
        agg_speedup = total_torch / total_asc if total_asc > 0 else 0.0
    else:
        geo_mean = 0.0
        total_asc = 0.0
        total_torch = 0.0
        agg_speedup = 0.0

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Valid test cases:     {len(valid)}/{len(results)}")
    print(f"  Geometric mean speedup: {geo_mean:.4f}x")
    print(f"  Aggregate speedup:      {agg_speedup:.4f}x")
    for r in results:
        sp = f"{r['speedup_vs_torch']:.4f}x" if r['speedup_vs_torch'] else "N/A"
        print(f"    {r['case_id']}: ascendc={r['ascendc_avg_us']:.1f}us  torch={r['torch_avg_us']:.1f}us  speedup={sp}")

    # Save benchmark JSON
    benchmark_json = {
        "op_name": "pre_split_mixes",
        "arch": "ascend910b2",
        "geometric_mean_speedup": round(geo_mean, 4),
        "aggregate_speedup": round(agg_speedup, 4),
        "total_ascendc_avg_us": round(total_asc, 2),
        "total_torch_avg_us": round(total_torch, 2),
        "valid_cases": len(valid),
        "total_cases": len(results),
        "per_case": results,
    }

    json_path = os.path.join(SCRIPT_DIR, "benchmark_result.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_json, f, indent=2)
    print(f"\nSaved: {json_path}")

    return benchmark_json


if __name__ == "__main__":
    main()
