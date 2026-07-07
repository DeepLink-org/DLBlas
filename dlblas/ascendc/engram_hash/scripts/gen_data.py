#!/usr/bin/env python3
"""
Generate input data + golden output for the engram_hash AscendC kernel.

Reuses origin/engram_hash.py (Model.forward, generate_test_data, make_offsets)
so the golden is byte-identical to the PyTorch reference. Inputs and golden are
dumped as raw binary under <op>/input and <op>/output; the direct-invoke
executable reads these same inputs and its output is compared bit-exact by
verify_result.py.

Usage:
  python3 gen_data.py --nt 4096 --ngram 3 --layers 2 --tables 8 [--seed 42]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')
from engram_hash import Model, generate_test_data  # noqa: E402

OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(OP_DIR, 'input')
OUTPUT_DIR = os.path.join(OP_DIR, 'output')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gen(nt, ngram, layers, tables, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    params = {'num_tokens': nt, 'ngram': ngram, 'layers': layers, 'tables': tables}
    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(params)

    model = Model()
    with torch.no_grad():
        golden = model.forward(ngram_token_ids, multipliers, vocab_sizes, offsets)

    P = ngram - 1
    W = P * tables
    assert list(golden.shape) == [layers, nt, W], \
        f"golden shape {list(golden.shape)} != [{layers},{nt},{W}]"
    assert golden.dtype == torch.int32, f"golden dtype {golden.dtype} != int32"

    # Dump inputs (row-major contiguous) as raw binary.
    ngram_token_ids.contiguous().numpy().astype(np.int32).tofile(
        os.path.join(INPUT_DIR, 'ngram_token_ids.bin'))
    multipliers.contiguous().numpy().astype(np.int64).tofile(
        os.path.join(INPUT_DIR, 'multipliers.bin'))
    vocab_sizes.contiguous().numpy().astype(np.int32).tofile(
        os.path.join(INPUT_DIR, 'vocab_sizes.bin'))
    offsets.contiguous().numpy().astype(np.int32).tofile(
        os.path.join(INPUT_DIR, 'offsets.bin'))

    # Dump golden.
    golden.contiguous().numpy().astype(np.int32).tofile(
        os.path.join(OUTPUT_DIR, 'golden.bin'))

    meta = {
        'nt': nt, 'ngram': ngram, 'layers': layers, 'tables': tables,
        'P': P, 'W': W, 'seed': seed,
        'out_shape': [layers, nt, W],
        'ngram_shape': [nt, ngram],
        'mult_shape': [layers, ngram],
        'vocab_shape': [layers, P, tables],
        'offsets_shape': [layers, W],
    }
    with open(os.path.join(INPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[gen_data] NT={nt} N={ngram} L={layers} T={tables} -> "
          f"out[{layers},{nt},{W}] golden bytes={golden.numel()*4}")
    print(f"[gen_data] inputs -> {INPUT_DIR}")
    print(f"[gen_data] golden -> {os.path.join(OUTPUT_DIR, 'golden.bin')}")
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nt', type=int, default=4096)
    ap.add_argument('--ngram', type=int, default=3)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--tables', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    gen(args.nt, args.ngram, args.layers, args.tables, args.seed)
