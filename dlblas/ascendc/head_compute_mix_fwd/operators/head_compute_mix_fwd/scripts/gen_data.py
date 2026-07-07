# ============================================================================
# Test data generation for head_compute_mix_fwd
#
# Generates binary files for direct-invoke testing:
#   input/input_mix.bin   - FP16, shape [batch, n1, mhc_mult]
#   input/mhc_scale.bin   - FP16, scalar
#   input/mhc_base.bin    - FP16, [mhc_mult]
#   input/mhc_pre_eps.bin - FP32, scalar
#   output/golden.bin     - FP16 golden reference output
#
# Usage:
#   python3 gen_data.py                    # Default shape
#   python3 gen_data.py 1 128 4            # Small test
#   python3 gen_data.py 16 16384 4 1       # Default shape with fixed seed
#   python3 gen_data.py 1 128 4 0 special  # Extreme value test
# ============================================================================

import numpy as np
import os
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Parse command line args
batch = int(sys.argv[1]) if len(sys.argv) > 1 else 16
n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
mhc_mult = int(sys.argv[3]) if len(sys.argv) > 3 else 4
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
mode = sys.argv[5] if len(sys.argv) > 5 else "normal"

np.random.seed(seed)

total_elements = batch * n1 * mhc_mult

if mode == "normal":
    input_mix = np.random.randn(batch, n1, mhc_mult).astype(np.float16)
    mhc_scale = np.random.randn(1).astype(np.float16)
    mhc_base = np.random.randn(mhc_mult).astype(np.float16)
    mhc_pre_eps = np.float32(0.01)
elif mode == "zeros":
    input_mix = np.zeros((batch, n1, mhc_mult), dtype=np.float16)
    mhc_scale = np.ones(1, dtype=np.float16)
    mhc_base = np.zeros(mhc_mult, dtype=np.float16)
    mhc_pre_eps = np.float32(0.01)
elif mode == "extreme":
    # Test sigmoid saturation regions
    input_mix = np.array([[[10.0, 5.0, -5.0, -10.0]]], dtype=np.float16)  # [1,1,4]
    mhc_scale = np.array([1.0], dtype=np.float16)
    mhc_base = np.zeros(4, dtype=np.float16)
    mhc_pre_eps = np.float32(0.0)
elif mode == "asymmetric":
    # Test different base values per channel
    input_mix = np.ones((batch, n1, mhc_mult), dtype=np.float16)
    mhc_scale = np.array([2.0], dtype=np.float16)
    mhc_base = np.array([0.1, 1.0, -0.5, -2.0], dtype=np.float16)
    mhc_pre_eps = np.float32(0.01)
elif mode == "large_pos":
    input_mix = np.ones((batch, n1, mhc_mult), dtype=np.float16) * 5.0
    mhc_scale = np.array([1.0], dtype=np.float16)
    mhc_base = np.zeros(mhc_mult, dtype=np.float16)
    mhc_pre_eps = np.float32(0.01)
elif mode == "large_neg":
    input_mix = np.ones((batch, n1, mhc_mult), dtype=np.float16) * (-5.0)
    mhc_scale = np.array([1.0], dtype=np.float16)
    mhc_base = np.zeros(mhc_mult, dtype=np.float16)
    mhc_pre_eps = np.float32(0.01)
else:
    raise ValueError(f"Unknown mode: {mode}")

# Write input files
input_mix.flatten().tofile("input/input_mix.bin")
mhc_scale.flatten().tofile("input/mhc_scale.bin")
mhc_base.flatten().tofile("input/mhc_base.bin")
mhc_pre_eps.tofile("input/mhc_pre_eps.bin")

# Compute golden
golden = compute_golden(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
golden_flat = golden.flatten() if hasattr(golden, 'flatten') else golden.reshape(-1)
golden_flat.tofile("output/golden.bin")

print(f"Generated test data: [{batch}, {n1}, {mhc_mult}], {total_elements} elements, mode={mode}")
print(f"  input/input_mix.bin: FP16, {total_elements} elements, {total_elements*2} bytes")
print(f"  input/mhc_scale.bin: FP16, 1 element")
print(f"  input/mhc_base.bin: FP16, {mhc_mult} elements")
print(f"  input/mhc_pre_eps.bin: FP32, 1 element")
print(f"  output/golden.bin: FP16, {total_elements} elements")
