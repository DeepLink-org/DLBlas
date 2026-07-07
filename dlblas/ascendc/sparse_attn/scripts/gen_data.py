# ----------------------------------------------------------------------------------------------------------
# gen_data.py - Test data generation for sparse_attn (corrected bf16 handling)
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import sparse_attn_ref

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def generate_case(b, m, n, h, d, topk, invalid=0.0, same_idx=False, seed=42):
    rng = np.random.RandomState(seed)
    softmax_scale = d ** -0.5

    # Generate in PyTorch with proper bf16
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h, dtype=torch.float32)

    if same_idx:
        idxs = torch.zeros(b, m, topk, dtype=torch.int32)
    else:
        idxs = torch.randint(0, n, (b, m, topk), dtype=torch.int32)

    if invalid > 0:
        total = b * m * topk
        n_inv = int(total * invalid)
        flat = idxs.reshape(-1)
        pos = torch.randperm(total)[:n_inv]
        flat[pos] = -1
        idxs = flat.reshape(b, m, topk)

    # Golden
    golden = sparse_attn_ref(q, kv, sink, idxs, softmax_scale)

    # Store as raw bytes: Q/KV as bf16 (int16), sink as fp32, idxs as int32, golden as bf16
    q.view(torch.int16).numpy().tofile("input/input_q.bin")
    kv.view(torch.int16).numpy().tofile("input/input_kv.bin")
    sink.numpy().tofile("input/input_sink.bin")
    idxs.numpy().tofile("input/input_idxs.bin")
    golden.view(torch.int16).numpy().tofile("output/golden.bin")

    print(f"  Q: {q.shape} bf16, KV: {kv.shape} bf16, sink: {sink.shape} fp32")
    print(f"  idxs: {idxs.shape} int32, invalid: {(idxs<0).sum().item()}/{idxs.numel()}")
    print(f"  Golden: {golden.shape} bf16, sum_abs={golden.float().abs().sum():.4f}")


CASES = [(1,2,16,32,8,64,16,0.0,False,42),(2,2,16,32,8,64,16,0.5,False,42),
         (3,2,16,32,8,64,16,1.0,False,42),(4,2,16,32,8,64,16,0.0,True,42),
         (5,1,1,32,4,32,8,0.0,False,42)]

cid = int(sys.argv[1]) if len(sys.argv)>=2 else 1
for tc in CASES:
    if tc[0]==cid or cid==0:
        print(f"TC-{tc[0]:02d}: b={tc[1]} m={tc[2]} n={tc[3]} h={tc[4]} d={tc[5]} topk={tc[6]}")
        generate_case(*tc[1:])
        if cid!=0: break
print("Done.")
