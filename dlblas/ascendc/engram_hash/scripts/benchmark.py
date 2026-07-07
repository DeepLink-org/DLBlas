#!/usr/bin/env python3
"""
Benchmark engram_hash: AscendC kernel latency vs PyTorch reference on NPU.

Reports per-shape latency and speedup, plus a geometric-mean speedup over all
shapes. Also runs a direct-invoke core-scaling probe (1 vs 48 cores) to confirm
the multi-core token split is effective (P0 optimization).

Usage:
  ASCEND_RT_VISIBLE_DEVICES=2 python3 scripts/benchmark.py
"""
import os
import sys
import time
import json
import subprocess
import numpy as np
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')
from engram_hash import Model, generate_test_data  # noqa: E402

OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(OP_DIR, 'output')
SO = os.path.join(OP_DIR, 'build', 'libengram_hash_ops.so')
EXE = os.path.join(OP_DIR, 'build', 'engram_hash_custom')
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.ops.load_library(SO)


def sync():
    torch.npu.synchronize()


def bench(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeat * 1000.0  # ms


def bench_shapes():
    shapes = [
        (32, 3, 2, 8),
        (256, 3, 2, 8),
        (1024, 3, 2, 8),
        (4096, 3, 2, 8),
        (65536, 3, 2, 8),
        (4096, 5, 4, 16),
        (256, 5, 4, 16),
    ]
    rows = []
    for nt, N, L, T in shapes:
        torch.manual_seed(42)
        ng, mu, vo, of = generate_test_data(
            {'num_tokens': nt, 'ngram': N, 'layers': L, 'tables': T})
        model = Model()
        ng_n = ng.to('npu:0'); mu_n = mu.to('npu:0')
        vo_n = vo.to('npu:0'); of_n = of.to('npu:0')

        # correctness sanity
        with torch.no_grad():
            g = model.forward(ng, mu, vo, of)
        a = torch.ops.npu.engram_hash(ng_n, mu_n, vo_n, of_n).cpu()
        exact = torch.equal(a, g)

        def torch_fn():
            with torch.no_grad():
                return model.forward(ng_n, mu_n, vo_n, of_n)

        def ascend_fn():
            return torch.ops.npu.engram_hash(ng_n, mu_n, vo_n, of_n)

        t_ms = bench(torch_fn)
        a_ms = bench(ascend_fn)
        sp = t_ms / a_ms if a_ms > 0 else 0.0
        P = N - 1; W = P * T
        rows.append({'nt': nt, 'N': N, 'L': L, 'T': T, 'W': W,
                     'exact': bool(exact),
                     'torch_ms': round(t_ms, 4),
                     'ascend_ms': round(a_ms, 4),
                     'speedup': round(sp, 4)})
        print(f"  NT={nt:6d} N={N} L={L} T={T:2d} W={W:3d}  "
              f"torch={t_ms:8.4f}ms  ascend={a_ms:8.4f}ms  speedup={sp:6.3f}x  exact={exact}")
    return rows


def core_scaling():
    """Direct-invoke wall-clock scaling: 1 core vs 48 cores at NT=65536."""
    print("\n--- Core scaling probe (direct invoke, NT=65536 N=3 L=2 T=8) ---")
    nt, N, L, T = 65536, 3, 2, 8
    torch.manual_seed(42)
    ng, mu, vo, of = generate_test_data(
        {'num_tokens': nt, 'ngram': N, 'layers': L, 'tables': T})
    inp = os.path.join(OP_DIR, 'input')
    ng.numpy().astype(np.int32).tofile(os.path.join(inp, 'ngram_token_ids.bin'))
    mu.numpy().astype(np.int64).tofile(os.path.join(inp, 'multipliers.bin'))
    vo.numpy().astype(np.int32).tofile(os.path.join(inp, 'vocab_sizes.bin'))
    of.numpy().astype(np.int32).tofile(os.path.join(inp, 'offsets.bin'))

    res = {}
    for cores in [1, 8, 24, 48]:
        env = dict(os.environ, EH_IO_DIR=OP_DIR)
        # warm + timed wall clock of the executable (includes fixed ACL overhead)
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            r = subprocess.run([EXE, str(nt), str(N), str(L), str(T), '0', str(cores)],
                               capture_output=True, text=True, env=env, timeout=300)
            dt = time.perf_counter() - t0
            best = min(best, dt)
            if r.returncode != 0:
                print(f"    cores={cores}: FAIL rc={r.returncode}")
                break
        res[cores] = round(best * 1000.0, 2)
        print(f"    cores={cores:2d}: wall={best*1000:.2f} ms (includes fixed ACL init)")
    return res


def main():
    print("=" * 62)
    print("engram_hash AscendC benchmark")
    print("=" * 62)
    rows = bench_shapes()

    speeds = [r['speedup'] for r in rows if r['exact'] and r['speedup'] > 0]
    geomean = float(np.exp(np.mean(np.log(speeds)))) if speeds else 0.0
    all_exact = all(r['exact'] for r in rows)
    print(f"\nGeomean speedup vs torch (bit-exact shapes): {geomean:.4f}x")
    print(f"All shapes bit-exact: {all_exact}")

    scaling = core_scaling()

    result = {
        'per_shape': rows,
        'geomean_speedup_vs_torch': round(geomean, 4),
        'all_bit_exact': all_exact,
        'core_scaling_wall_ms': scaling,
    }
    out_path = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    return result


if __name__ == '__main__':
    main()
