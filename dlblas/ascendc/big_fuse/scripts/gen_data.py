# ============================================================================
# Test data generation for big_fuse operator
# ============================================================================

import numpy as np
import os

# Constants
N_TOKENS = 512
MHC_MULT = 4
HIDDEN_SIZE = 1280
RGS = MHC_MULT * HIDDEN_SIZE  # 5120
MHC_MULT3 = 2 * MHC_MULT + MHC_MULT * MHC_MULT  # 24

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Generate input data
# residual: [1, 512, 4, 1280] bf16
# Using moderate values to avoid extreme sigmoid outputs
residual = np.random.randn(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE).astype(np.float32) * 0.5

# fn: [24, 5120] fp32 (weight matrix, small init like PyTorch)
fn = np.random.randn(MHC_MULT3, RGS).astype(np.float32) * 1e-4

# mhc_scale: [3] fp32
mhc_scale = np.random.randn(3).astype(np.float32) * 0.1

# mhc_base: [24] fp32
mhc_base = np.random.randn(MHC_MULT3).astype(np.float32) * 0.1

# Save as raw binary
# residual: store as bf16 (convert via uint16)
# For bf16, we truncate mantissa from fp32
# Simple approach: view fp32 as uint32, shift right 16 bits, store as uint16
# Actually, bf16 just truncates the lower 16 bits of fp32
residual_f32 = residual.astype(np.float32)
residual_u32 = residual_f32.view(np.uint32)
residual_bf16 = (residual_u32 >> 16).astype(np.uint16)

residual_bf16.tofile("input/residual.bin")
fn.tofile("input/fn.bin")
mhc_scale.tofile("input/mhc_scale.bin")
mhc_base.tofile("input/mhc_base.bin")

print(f"Generated test data:")
print(f"  residual:   {residual.shape}, dtype=bfloat16, values in [{residual_f32.min():.4f}, {residual_f32.max():.4f}]")
print(f"  fn:         {fn.shape}, dtype=float32")
print(f"  mhc_scale:  {mhc_scale.shape}, dtype=float32, values={mhc_scale}")
print(f"  mhc_base:   {mhc_base.shape}, dtype=float32")
print(f"  Files written to input/")
