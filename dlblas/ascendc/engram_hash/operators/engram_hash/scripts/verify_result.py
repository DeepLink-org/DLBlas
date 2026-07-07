#!/usr/bin/env python3
"""
Bit-exact verification for engram_hash: compare kernel output (output/out.bin)
against the PyTorch golden (output/golden.bin) with np.array_equal (atol=0).

Integer-compute operator → the pass criterion is exact binary equality.

Usage:
  python3 verify_result.py            # uses input/meta.json for shape
"""
import os
import sys
import json
import numpy as np

OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(OP_DIR, 'input')
OUTPUT_DIR = os.path.join(OP_DIR, 'output')


def main():
    meta_path = os.path.join(INPUT_DIR, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"[verify] FAIL: {meta_path} missing (run gen_data.py)")
        return 1
    with open(meta_path) as f:
        meta = json.load(f)

    L, NT, W = meta['out_shape']
    golden_path = os.path.join(OUTPUT_DIR, 'golden.bin')
    out_path = os.path.join(OUTPUT_DIR, 'out.bin')
    if not os.path.exists(golden_path):
        print(f"[verify] FAIL: {golden_path} missing")
        return 1
    if not os.path.exists(out_path):
        print(f"[verify] FAIL: {out_path} missing (run the kernel first)")
        return 1

    golden = np.fromfile(golden_path, dtype=np.int32)
    out = np.fromfile(out_path, dtype=np.int32)

    expected = L * NT * W
    if golden.size != expected:
        print(f"[verify] FAIL: golden size {golden.size} != {expected}")
        return 1
    if out.size != expected:
        print(f"[verify] FAIL: out size {out.size} != {expected}")
        return 1

    golden = golden.reshape(L, NT, W)
    out = out.reshape(L, NT, W)

    if np.array_equal(golden, out):
        print(f"[verify] PASS (bit-exact): shape=[{L},{NT},{W}], elems={expected}")
        return 0

    diff = out != golden
    nmis = int(diff.sum())
    idx = np.argwhere(diff)
    print(f"[verify] FAIL: {nmis}/{expected} elements differ")
    for k in range(min(10, idx.shape[0])):
        l, t, w = idx[k]
        print(f"    [{l},{t},{w}] golden={golden[l,t,w]} out={out[l,t,w]}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
