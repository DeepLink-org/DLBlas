#!/usr/bin/env python3
# ============================================================================
# Multi-level test runner for big_fuse operator
# Runs Level 0 (small values), Level 1 (standard), Level 2 (extreme/boundary)
# ============================================================================

import numpy as np
import os
import sys
import subprocess

N_TOKENS = 512
MHC_MULT = 4
HIDDEN_SIZE = 1280
RGS = MHC_MULT * HIDDEN_SIZE  # 5120
MHC_MULT3 = 2 * MHC_MULT + MHC_MULT * MHC_MULT  # 24

RMS_EPS = 1e-6
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT_VALUE = 1.0
SINKHORN_REPEAT = 10

# Precision thresholds
FP32_MERE_THRESH = 2.0 ** (-10)  # 0.000977
BF16_MERE_THRESH = 2.0 ** (-7)   # 0.00781
BF16_MAX_ABS_THRESH = 2.0 ** (-6)  # 0.015625

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "build")


def make_bf16(arr_f32):
    """Convert fp32 array to bf16 uint16 representation."""
    u32 = arr_f32.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def load_bf16_as_f32(filepath, shape):
    """Load bf16 binary and convert to fp32."""
    data = np.fromfile(filepath, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    return data_u32.view(np.float32).reshape(shape)


def compute_golden(residual_bf16_u16, fn_f32, mhc_scale_f32, mhc_base_f32):
    """PyTorch golden computation (pure numpy, no torch dependency for CI)."""
    import torch

    # Load residual from bf16
    residual_u32 = residual_bf16_u16.astype(np.uint32) << 16
    residual_f32 = residual_u32.view(np.float32).reshape(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE)
    residual_t = torch.from_numpy(residual_f32.copy()).bfloat16()

    fn_t = torch.from_numpy(fn_f32.reshape(MHC_MULT3, RGS).copy())
    scale_t = torch.from_numpy(mhc_scale_f32.copy())
    base_t = torch.from_numpy(mhc_base_f32.copy())

    # S1: RMS-normed linear projection
    n0, n1 = residual_t.shape[:2]
    x = residual_t.flatten(2, 3).float().reshape(n0 * n1, -1)
    mixes = x @ fn_t.T
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * (sqrsum / x.shape[-1] + RMS_EPS).rsqrt()
    mixes = mixes.view(n0, n1, -1)

    # S2: Split mixing logits
    a, b = mixes.shape[:2]
    scale_exp = torch.cat([
        scale_t[0].expand(MHC_MULT),
        scale_t[1].expand(MHC_MULT),
        scale_t[2].expand(MHC_MULT * MHC_MULT),
    ])
    mixes = mixes * scale_exp + base_t
    pre_mix = mixes[:, :, :MHC_MULT].sigmoid().unsqueeze(-1) + MHC_PRE_EPS
    post_mix = (mixes[:, :, MHC_MULT:2*MHC_MULT].sigmoid() * MHC_POST_MULT_VALUE).unsqueeze(-1)
    comb_mix = mixes[:, :, 2*MHC_MULT:].view(a, b, MHC_MULT, MHC_MULT)

    # S3: Sinkhorn
    eps = MHC_SINKHORN_EPS
    C = comb_mix.softmax(-1) + eps
    C = C / (C.sum(-2, keepdim=True) + eps)
    for _ in range(SINKHORN_REPEAT - 1):
        C = C / (C.sum(-1, keepdim=True) + eps)
        C = C / (C.sum(-2, keepdim=True) + eps)

    # S4: Apply mix
    layer_input = (residual_t * pre_mix).sum(-2).bfloat16()

    post_out = post_mix.squeeze(-1).reshape(N_TOKENS, MHC_MULT).float().cpu().numpy()
    comb_out = C.reshape(N_TOKENS, MHC_MULT, MHC_MULT).float().cpu().numpy()
    li_f32 = layer_input.reshape(N_TOKENS, HIDDEN_SIZE).float().cpu().numpy()

    return post_out, comb_out, li_f32


def generate_data(level, seed):
    """Generate test data for a given level."""
    np.random.seed(seed)

    if level == 0:
        # Level 0: Very small values (near-zero)
        residual = np.random.randn(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE).astype(np.float32) * 0.001
        fn = np.random.randn(MHC_MULT3, RGS).astype(np.float32) * 1e-6
        mhc_scale = np.array([0.01, -0.005, 0.008], dtype=np.float32)
        mhc_base = np.random.randn(MHC_MULT3).astype(np.float32) * 0.001
    elif level == 2:
        # Level 2: Extreme values (large magnitudes)
        residual = np.random.randn(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE).astype(np.float32) * 2.0
        fn = np.random.randn(MHC_MULT3, RGS).astype(np.float32) * 1e-2
        mhc_scale = np.array([0.5, -0.3, 0.4], dtype=np.float32)
        mhc_base = np.random.randn(MHC_MULT3).astype(np.float32) * 0.5
    elif level == 3:
        # Level 3: Zero inputs (test boundary conditions)
        residual = np.zeros((1, N_TOKENS, MHC_MULT, HIDDEN_SIZE), dtype=np.float32)
        fn = np.random.randn(MHC_MULT3, RGS).astype(np.float32) * 1e-4
        mhc_scale = np.ones(3, dtype=np.float32)
        mhc_base = np.zeros(MHC_MULT3, dtype=np.float32)
    else:
        # Level 1: Standard values (default)
        residual = np.random.randn(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE).astype(np.float32) * 0.5
        fn = np.random.randn(MHC_MULT3, RGS).astype(np.float32) * 1e-4
        mhc_scale = np.random.randn(3).astype(np.float32) * 0.1
        mhc_base = np.random.randn(MHC_MULT3).astype(np.float32) * 0.1

    return residual, fn, mhc_scale, mhc_base


def verify_output(name, ascend_path, golden_path, shape, dtype, mere_thresh):
    """Verify output against golden."""
    if dtype == np.float32:
        ascend = np.fromfile(ascend_path, dtype=np.float32).reshape(shape)
        golden = np.fromfile(golden_path, dtype=np.float32).reshape(shape)
    elif dtype == 'bf16':
        ascend = load_bf16_as_f32(ascend_path, shape)
        golden = load_bf16_as_f32(golden_path, shape)
    else:
        return False, f"Unsupported dtype: {dtype}"

    a_flat = ascend.flatten()
    g_flat = golden.flatten()

    abs_diff = np.abs(a_flat - g_flat)
    denom = np.maximum(np.maximum(np.abs(a_flat), np.abs(g_flat)), 1e-10)
    mere = np.max(abs_diff / denom)
    max_abs_err = np.max(abs_diff)

    nan_count = np.sum(np.isnan(a_flat))
    inf_count = np.sum(np.isinf(a_flat))

    if nan_count > 0 or inf_count > 0:
        return False, f"{name}: NaN={nan_count}, Inf={inf_count}"

    if dtype == 'bf16':
        passed = max_abs_err < BF16_MAX_ABS_THRESH or mere < mere_thresh
    else:
        passed = mere < mere_thresh

    info = f"{name}: MERE={mere:.6e}, max_abs_err={max_abs_err:.6e}"
    return passed, info


def run_level(level, seed):
    """Run a single test level."""
    level_names = {0: "Level 0 (near-zero)", 1: "Level 1 (standard)",
                   2: "Level 2 (extreme)", 3: "Level 3 (zero-input)"}
    print(f"\n{'='*60}")
    print(f"  {level_names.get(level, f'Level {level}')}  (seed={seed})")
    print(f"{'='*60}")

    residual, fn, mhc_scale, mhc_base = generate_data(level, seed)

    # Save inputs
    os.makedirs(os.path.join(BUILD_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(BUILD_DIR, "output"), exist_ok=True)

    residual_bf16 = make_bf16(residual)
    residual_bf16.tofile(os.path.join(BUILD_DIR, "input", "residual.bin"))
    fn.tofile(os.path.join(BUILD_DIR, "input", "fn.bin"))
    mhc_scale.tofile(os.path.join(BUILD_DIR, "input", "mhc_scale.bin"))
    mhc_base.tofile(os.path.join(BUILD_DIR, "input", "mhc_base.bin"))

    # Compute golden
    post_golden, comb_golden, li_golden = compute_golden(
        residual_bf16, fn, mhc_scale, mhc_base
    )

    post_golden.tofile(os.path.join(BUILD_DIR, "output", "post_mix_golden.bin"))
    comb_golden.tofile(os.path.join(BUILD_DIR, "output", "comb_mix_golden.bin"))
    # layer_input golden in bf16 uint16
    li_u32 = li_golden.view(np.uint32)
    (li_u32 >> 16).astype(np.uint16).tofile(
        os.path.join(BUILD_DIR, "output", "layer_input_golden.bin"))

    # Run kernel
    exe = os.path.join(BUILD_DIR, "big_fuse")
    result = subprocess.run([exe], capture_output=True, text=True, cwd=BUILD_DIR)
    if result.returncode != 0:
        print(f"  Kernel FAILED: {result.stderr[-500:]}")
        return False

    # Verify
    all_pass = True
    for name, fname, shape, dtype, thresh in [
        ("post_mix", "post_mix.bin", (N_TOKENS, MHC_MULT), np.float32, FP32_MERE_THRESH),
        ("comb_mix", "comb_mix.bin", (N_TOKENS, MHC_MULT, MHC_MULT), np.float32, FP32_MERE_THRESH),
        ("layer_input", "layer_input.bin", (N_TOKENS, HIDDEN_SIZE), 'bf16', BF16_MERE_THRESH),
    ]:
        passed, info = verify_output(
            name,
            os.path.join(BUILD_DIR, "output", fname),
            os.path.join(BUILD_DIR, "output", f"{name}_golden.bin"),
            shape, dtype, thresh
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {info}")
        all_pass = all_pass and passed

    return all_pass


def main():
    # Must build first
    if not os.path.exists(os.path.join(BUILD_DIR, "big_fuse")):
        print("ERROR: Kernel not built. Run 'bash run.sh' first.")
        return 1

    results = {}
    for level in [0, 1, 2, 3]:
        results[level] = run_level(level, seed=42 + level)

    print(f"\n{'='*60}")
    print("  MULTI-LEVEL TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for level, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_pass = False
        level_names = {0: "Level 0 (near-zero)", 1: "Level 1 (standard)",
                       2: "Level 2 (extreme)", 3: "Level 3 (zero-input)"}
        print(f"  [{status}] {level_names.get(level, f'Level {level}')}")

    print(f"{'='*60}")
    if all_pass:
        print("  ALL LEVELS PASSED!")
    else:
        print("  SOME LEVELS FAILED!")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
