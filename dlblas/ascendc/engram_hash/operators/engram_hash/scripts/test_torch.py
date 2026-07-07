#!/usr/bin/env python3
"""
PyTorch integration test for engram_hash: exercise torch.ops.npu.engram_hash
on the NPU and compare bit-exact against the origin Model.forward reference.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=2 python3 scripts/test_torch.py
"""
import os
import sys
import numpy as np
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')
from engram_hash import Model, generate_test_data  # noqa: E402

OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = os.path.join(OP_DIR, 'build', 'libengram_hash_ops.so')

torch.ops.load_library(SO)
print(f"[torch] loaded {SO}")


def run(nt, ngram, layers, tables, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ng, mu, vo, of = generate_test_data(
        {'num_tokens': nt, 'ngram': ngram, 'layers': layers, 'tables': tables})

    with torch.no_grad():
        golden = Model().forward(ng, mu, vo, of)

    ng_n = ng.to('npu:0')
    mu_n = mu.to('npu:0')
    vo_n = vo.to('npu:0')
    of_n = of.to('npu:0')

    out = torch.ops.npu.engram_hash(ng_n, mu_n, vo_n, of_n)
    out_cpu = out.cpu()

    ok = torch.equal(out_cpu, golden)
    P = ngram - 1
    W = P * tables
    tag = "PASS" if ok else "FAIL"
    print(f"[torch] NT={nt} N={ngram} L={layers} T={tables} "
          f"out={list(out_cpu.shape)} dtype={out_cpu.dtype} bit-exact={tag}")
    if not ok:
        diff = (out_cpu != golden)
        print(f"        mismatches={int(diff.sum())}/{golden.numel()}")
        idx = torch.nonzero(diff)[:5]
        for x in idx:
            l, t, w = x.tolist()
            print(f"        [{l},{t},{w}] golden={golden[l,t,w]} out={out_cpu[l,t,w]}")
    return ok


def main():
    cases = [
        (4096, 3, 2, 8),   # baseline
        (32, 3, 2, 8),     # tokens < cores
        (256, 5, 4, 16),   # big W, long chain
        (256, 3, 2, 1),    # W=1, multi-layer (alignment edge)
        (4096, 2, 1, 1),   # minimal W
        (65536, 3, 2, 8),  # large batch multi-tile
        (1000, 4, 2, 4),   # arbitrary
    ]
    allok = True
    for c in cases:
        allok &= run(*c)
    print("\n[torch] ALL PASS" if allok else "\n[torch] SOME FAILED")
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
