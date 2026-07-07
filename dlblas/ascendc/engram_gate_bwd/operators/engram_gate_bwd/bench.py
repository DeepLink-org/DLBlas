#!/usr/bin/env python3
"""Quick benchmark: engram_gate_bwd AscendC vs PyTorch"""
import time, subprocess, os, sys
import numpy as np
sys.path.insert(0, "scripts")
from golden import compute_golden

WD = "/mnt/data01/zmz/workspace/12agent/waic/build/engram_gate_bwd/operators/engram_gate_bwd"

def bench_torch(T, H, D, iters=100):
    np.random.seed(42)
    go = np.random.randn(T, H, D).astype(np.float32)
    x  = np.random.randn(T, H, D).astype(np.float32)
    k  = np.random.randn(T, H, D).astype(np.float32)
    v  = np.random.randn(T, D).astype(np.float32)
    wh = np.random.randn(H, D).astype(np.float32)
    we = np.random.randn(H, D).astype(np.float32)
    # warmup
    for _ in range(10): compute_golden(go, x, k, v, wh, we)
    t0 = time.perf_counter()
    for _ in range(iters): compute_golden(go, x, k, v, wh, we)
    return (time.perf_counter() - t0) / iters * 1e6

def bench_ascend(T, H, D, iters=5):
    os.chdir(WD)
    subprocess.run(["python3", "scripts/gen_data.py"], cwd="build", capture_output=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        subprocess.run(["./engram_gate_bwd", str(T), str(H), str(D), "1e-6", "1e-20"],
                       cwd="build", capture_output=True)
    return (time.perf_counter() - t0) / iters * 1e6

if __name__ == "__main__":
    os.chdir(WD)
    T, H, D = 14, 4, 128
    tu = bench_torch(T, H, D)
    print(f"PyTorch(numpy): {tu:.1f} us")
    au = bench_ascend(T, H, D)
    print(f"AscendC:        {au:.1f} us")
    print(f"Speedup:        {tu/au:.2f}x")
    print(f"RESULT: torch={tu:.1f} ascend={au:.1f} speedup={tu/au:.2f}")
