#!/usr/bin/env python3
"""
Full verification matrix for engram_hash: sweep shapes x core counts, run the
direct-invoke executable on NPU, and check bit-exact against PyTorch golden.

For each (NT,N,L,T,cores):
  1. gen_data.py generates inputs + golden for the shape
  2. engram_hash_custom runs the kernel with the given core count
  3. verify_result.py checks np.array_equal (atol=0)

Usage:
  ASCEND_RT_VISIBLE_DEVICES=2 python3 scripts/run_verify_matrix.py [--smoke]
"""
import os
import sys
import json
import subprocess
import argparse
import itertools
import numpy as np

sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')
from engram_hash import Model, generate_test_data  # noqa: E402
import torch  # noqa: E402

OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(OP_DIR, 'input')
OUTPUT_DIR = os.path.join(OP_DIR, 'output')
EXE = os.path.join(OP_DIR, 'build', 'engram_hash_custom')


def make_case(nt, ngram, layers, tables, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = {'num_tokens': nt, 'ngram': ngram, 'layers': layers, 'tables': tables}
    ng, mu, vo, of = generate_test_data(params)
    with torch.no_grad():
        golden = Model().forward(ng, mu, vo, of)
    P = ngram - 1
    W = P * tables
    ng.contiguous().numpy().astype(np.int32).tofile(os.path.join(INPUT_DIR, 'ngram_token_ids.bin'))
    mu.contiguous().numpy().astype(np.int64).tofile(os.path.join(INPUT_DIR, 'multipliers.bin'))
    vo.contiguous().numpy().astype(np.int32).tofile(os.path.join(INPUT_DIR, 'vocab_sizes.bin'))
    of.contiguous().numpy().astype(np.int32).tofile(os.path.join(INPUT_DIR, 'offsets.bin'))
    golden.contiguous().numpy().astype(np.int32).tofile(os.path.join(OUTPUT_DIR, 'golden.bin'))
    meta = {'nt': nt, 'ngram': ngram, 'layers': layers, 'tables': tables,
            'P': P, 'W': W, 'out_shape': [layers, nt, W]}
    with open(os.path.join(INPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    return golden.contiguous().numpy().astype(np.int32).reshape(layers, nt, W)


def run_case(nt, ngram, layers, tables, cores, golden):
    env = dict(os.environ)
    env['EH_IO_DIR'] = OP_DIR
    r = subprocess.run([EXE, str(nt), str(ngram), str(layers), str(tables), '0', str(cores)],
                       capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        return False, f"exe rc={r.returncode}: {r.stdout[-300:]} {r.stderr[-300:]}"
    out_path = os.path.join(OUTPUT_DIR, 'out.bin')
    out = np.fromfile(out_path, dtype=np.int32)
    if out.size != golden.size:
        return False, f"size {out.size} != {golden.size}"
    out = out.reshape(golden.shape)
    if np.array_equal(out, golden):
        return True, "bit-exact"
    nmis = int((out != golden).sum())
    return False, f"{nmis} mismatches"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='small subset only')
    args = ap.parse_args()

    if not os.path.exists(EXE):
        print(f"[matrix] FAIL: executable {EXE} not built")
        return 1

    if args.smoke:
        cases = [
            (4096, 3, 2, 8, 48),
            (32, 3, 2, 8, 48),
            (256, 5, 4, 16, 8),
        ]
    else:
        NTs = [32, 256, 1024, 4096, 65536]
        Ns = [2, 3, 4, 5]
        Ls = [1, 2, 4]
        Ts = [1, 4, 8, 16]
        cores_list = [1, 8, 24, 48]
        # Full cartesian product is large; use a representative sweep:
        #  (a) full NxLxT grid at a fixed medium NT and full cores each
        #  (b) NT sweep at baseline N,L,T over all core counts
        #  (c) explicit edge/regression cases
        cases = []
        # (a) shape grid at NT=256, cores=48
        for N, L, T in itertools.product(Ns, Ls, Ts):
            cases.append((256, N, L, T, 48))
        # (b) NT sweep x cores at baseline N=3,L=2,T=8
        for NT in NTs:
            for c in cores_list:
                cases.append((NT, 3, 2, 8, c))
        # (c) regression / edge
        cases += [
            (65536, 3, 2, 8, 48),   # large batch, multi-tile
            (32, 5, 4, 16, 48),     # tokens < cores, big W
            (100, 3, 2, 8, 48),     # tail not divisible
            (4096, 2, 1, 1, 48),    # minimal W=1
            (4096, 5, 4, 16, 48),   # max W
            (1000, 4, 2, 4, 24),    # arbitrary
        ]
        # dedup
        seen = set(); uniq = []
        for c in cases:
            if c not in seen:
                seen.add(c); uniq.append(c)
        cases = uniq

    npass = 0
    fails = []
    for i, (nt, ngram, layers, tables, cores) in enumerate(cases):
        golden = make_case(nt, ngram, layers, tables)
        ok, msg = run_case(nt, ngram, layers, tables, cores, golden)
        tag = "PASS" if ok else "FAIL"
        print(f"[{i+1}/{len(cases)}] NT={nt} N={ngram} L={layers} T={tables} cores={cores}: {tag} ({msg})")
        if ok:
            npass += 1
        else:
            fails.append((nt, ngram, layers, tables, cores, msg))

    print(f"\n[matrix] {npass}/{len(cases)} passed")
    if fails:
        print("[matrix] FAILURES:")
        for f in fails:
            print("   ", f)
        return 1
    print("[matrix] ALL BIT-EXACT")
    return 0


if __name__ == '__main__':
    sys.exit(main())
