# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Test data generation for engram_gate_bwd
# ============================================================================

import numpy as np
import os
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Default dimensions
T = 14
H = 4
D = 128
clamp_value = 1e-6
eps = 1e-20

# Override from command line
if len(sys.argv) >= 4:
    T = int(sys.argv[1])
    H = int(sys.argv[2])
    D = int(sys.argv[3])
if len(sys.argv) >= 5:
    clamp_value = float(sys.argv[4])
if len(sys.argv) >= 6:
    eps = float(sys.argv[5])

print(f"Generating test data: T={T}, H={H}, D={D}, cv={clamp_value}, eps={eps}")

np.random.seed(42)

# Generate bf16 data (pack as uint16 = 2 bytes per element)
def to_bf16(arr):
    """Convert f32 to bf16 by truncating mantissa, return as f32 with bf16 precision"""
    i32 = arr.view(np.uint32)
    i32 = i32 + 0x8000  # round to nearest even
    i32 = i32 & 0xFFFF0000
    return i32.view(np.float32)

def pack_bf16(arr):
    """Pack f32 array to bf16 as uint16 values (for binary I/O)"""
    i32 = arr.view(np.uint32)
    return (i32 >> 16).astype(np.uint16)

go  = to_bf16(np.random.randn(T, H, D).astype(np.float32) * 0.1)
x   = to_bf16(np.random.randn(T, H, D).astype(np.float32) * 0.1)
k   = to_bf16(np.random.randn(T, H, D).astype(np.float32) * 0.1)
v   = to_bf16(np.random.randn(T, D).astype(np.float32) * 0.1)
wh  = to_bf16(np.random.randn(H, D).astype(np.float32) * 0.1)
we  = to_bf16(np.random.randn(H, D).astype(np.float32) * 0.1)

# Save as packed bf16 (uint16, 2 bytes per element)
pack_bf16(go).tofile("input/input_go.bin")
pack_bf16(x).tofile("input/input_x.bin")
pack_bf16(k).tofile("input/input_k.bin")
pack_bf16(v).tofile("input/input_v.bin")
pack_bf16(wh).tofile("input/input_wh.bin")
pack_bf16(we).tofile("input/input_we.bin")

# Compute golden (f32)
golden = compute_golden(go, x, k, v, wh, we, clamp_value, eps)
names = ["grad_x", "grad_k", "grad_v", "grad_wh", "grad_we"]

for name, g in zip(names, golden):
    g.astype(np.float32).tofile(f"output/golden_{name}.bin")

print(f"Generated test data: T={T} H={H} D={D}")
print(f"  go:  {go.shape}, {go.dtype}")
print(f"  x:   {x.shape}, {x.dtype}")
print(f"  k:   {k.shape}, {k.dtype}")
print(f"  v:   {v.shape}, {v.dtype}")
print(f"  wh:  {wh.shape}, {wh.dtype}")
print(f"  we:  {we.shape}, {we.dtype}")
for name, g in zip(names, golden):
    print(f"  golden_{name}: {g.shape}, {g.dtype}")
