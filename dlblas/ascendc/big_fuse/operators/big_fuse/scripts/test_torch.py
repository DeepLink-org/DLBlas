#!/usr/bin/env python3
"""
PyTorch TORCH_LIBRARY access test for big_fuse operator.
Loads libbig_fuse_ops.so and calls torch.ops.npu.big_fuse().
Verifies output against PyTorch golden reference.
"""

import torch
import torch_npu
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_DIR = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(OP_DIR, "build")

# Constants
N_TOKENS = 512
MHC_MULT = 4
HIDDEN_SIZE = 1280
RGS = MHC_MULT * HIDDEN_SIZE
MHC_MULT3 = 2 * MHC_MULT + MHC_MULT * MHC_MULT

RMS_EPS = 1e-6
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT_VALUE = 1.0
SINKHORN_REPEAT = 10


def load_bf16_binary(filepath, shape):
    """Load bf16 binary file as torch tensor."""
    data = np.fromfile(filepath, dtype=np.uint16)
    data_u32 = data.astype(np.uint32) << 16
    data_f32 = data_u32.view(np.float32).reshape(shape)
    return torch.from_numpy(data_f32.copy()).bfloat16()


def compute_golden_pytorch(residual, fn, mhc_scale, mhc_base):
    """PyTorch reference golden (same as golden.py)."""
    # S1
    x = residual.flatten(2, 3).float().reshape(1, N_TOKENS, -1)
    fn_t = fn.view(MHC_MULT3, RGS)
    mixes = x @ fn_t.T
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * (sqrsum / x.shape[-1] + RMS_EPS).rsqrt()
    mixes = mixes.view(1, N_TOKENS, -1)

    # S2
    scale = torch.cat([
        mhc_scale[0].expand(MHC_MULT),
        mhc_scale[1].expand(MHC_MULT),
        mhc_scale[2].expand(MHC_MULT * MHC_MULT),
    ])
    mixes_b = mixes * scale + mhc_base
    pre_mix = mixes_b[:, :, :MHC_MULT].sigmoid().unsqueeze(-1) + MHC_PRE_EPS
    post_mix = (mixes_b[:, :, MHC_MULT:2*MHC_MULT].sigmoid() * MHC_POST_MULT_VALUE).unsqueeze(-1)
    comb_mix = mixes_b[:, :, 2*MHC_MULT:].view(1, N_TOKENS, MHC_MULT, MHC_MULT)

    # S3
    C = comb_mix.softmax(-1) + MHC_SINKHORN_EPS
    C = C / (C.sum(-2, keepdim=True) + MHC_SINKHORN_EPS)
    for _ in range(SINKHORN_REPEAT - 1):
        C = C / (C.sum(-1, keepdim=True) + MHC_SINKHORN_EPS)
        C = C / (C.sum(-2, keepdim=True) + MHC_SINKHORN_EPS)

    # S4
    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, C, layer_input


def verify(name, ascend_val, golden_val, dtype, mere_thresh):
    """Verify ascend output against golden."""
    a_flat = ascend_val.detach().cpu().float().flatten().numpy()
    g_flat = golden_val.detach().cpu().float().flatten().numpy()

    abs_diff = np.abs(a_flat - g_flat)
    denom = np.maximum(np.maximum(np.abs(a_flat), np.abs(g_flat)), 1e-10)
    mere = np.max(abs_diff / denom)
    max_abs_err = np.max(abs_diff)

    nan_count = np.sum(np.isnan(a_flat))
    inf_count = np.sum(np.isinf(a_flat))

    if nan_count > 0 or inf_count > 0:
        return False, f"NaN={nan_count}, Inf={inf_count}"

    if dtype == 'bf16':
        BF16_MAX_ABS = 2.0 ** (-6)
        passed = max_abs_err < BF16_MAX_ABS or mere < mere_thresh
    else:
        passed = mere < mere_thresh

    info = f"MERE={mere:.6e}, max_abs_err={max_abs_err:.6e}, {'PASS' if passed else 'FAIL'}"
    return passed, info


def main():
    print("=" * 60)
    print("big_fuse PyTorch TORCH_LIBRARY Access Test")
    print("=" * 60)

    # 1. Load inputs
    print("\n[1/4] Loading inputs...")
    residual_u16 = np.fromfile(
        os.path.join(OP_DIR, "input", "residual.bin"), dtype=np.uint16
    ).reshape(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE)
    residual_u32 = residual_u16.astype(np.uint32) << 16
    residual_f32 = residual_u32.view(np.float32)
    residual_npu = torch.from_numpy(residual_f32.copy()).bfloat16().npu()

    fn = np.fromfile(os.path.join(OP_DIR, "input", "fn.bin"), dtype=np.float32)
    fn_npu = torch.from_numpy(fn.reshape(MHC_MULT3, RGS).copy()).npu()

    mhc_scale = np.fromfile(os.path.join(OP_DIR, "input", "mhc_scale.bin"), dtype=np.float32)
    mhc_scale_npu = torch.from_numpy(mhc_scale.copy()).npu()

    mhc_base = np.fromfile(os.path.join(OP_DIR, "input", "mhc_base.bin"), dtype=np.float32)
    mhc_base_npu = torch.from_numpy(mhc_base.copy()).npu()
    print("  Inputs loaded and placed on NPU.")

    # 2. Load extension library
    print("\n[2/4] Loading libbig_fuse_ops.so...")
    so_path = os.path.join(BUILD_DIR, "libbig_fuse_ops.so")
    if not os.path.exists(so_path):
        print(f"  ERROR: {so_path} not found!")
        return 1
    torch.ops.load_library(so_path)
    print("  Library loaded.")

    # 3. Call operator
    print("\n[3/4] Calling torch.ops.npu.big_fuse()...")
    try:
        post_mix, comb_mix, layer_input = torch.ops.npu.big_fuse(
            residual_npu, fn_npu, mhc_scale_npu, mhc_base_npu
        )
        print(f"  post_mix:    {post_mix.shape}, dtype={post_mix.dtype}")
        print(f"  comb_mix:    {comb_mix.shape}, dtype={comb_mix.dtype}")
        print(f"  layer_input: {layer_input.shape}, dtype={layer_input.dtype}")
        print("  Operator call succeeded!")
    except Exception as e:
        print(f"  ERROR calling operator: {e}")
        return 1

    # 4. Verify against golden
    print("\n[4/4] Verifying against PyTorch golden...")
    residual_cpu = torch.from_numpy(residual_f32.copy()).bfloat16()
    fn_cpu = torch.from_numpy(fn.reshape(MHC_MULT3, RGS).copy())
    mhc_scale_cpu = torch.from_numpy(mhc_scale.copy())
    mhc_base_cpu = torch.from_numpy(mhc_base.copy())

    g_post, g_comb, g_layer = compute_golden_pytorch(
        residual_cpu, fn_cpu, mhc_scale_cpu, mhc_base_cpu
    )

    FP32_THRESH = 2.0 ** (-10)
    BF16_THRESH = 2.0 ** (-7)

    all_pass = True
    for name, ascend_val, golden_val, dtype, thresh in [
        ("post_mix", post_mix, g_post.squeeze(-1).reshape(N_TOKENS, MHC_MULT).float(), 'fp32', FP32_THRESH),
        ("comb_mix", comb_mix, g_comb.reshape(N_TOKENS, MHC_MULT, MHC_MULT).float(), 'fp32', FP32_THRESH),
        ("layer_input", layer_input, g_layer.reshape(N_TOKENS, HIDDEN_SIZE), 'bf16', BF16_THRESH),
    ]:
        passed, info = verify(name, ascend_val, golden_val, dtype, thresh)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {info}")
        if not passed:
            all_pass = False

    print()
    print("=" * 60)
    if all_pass:
        print("PyTorch TORCH_LIBRARY access test: ALL PASSED!")
    else:
        print("PyTorch TORCH_LIBRARY access test: SOME FAILED!")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
