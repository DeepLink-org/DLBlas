# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# PyTorch 通路测试 - MHC Post
#
# Compares the mhc_post custom NPU op with the reference implementation
# (mhc_post.py Model.forward) using bf16 inputs.
# ============================================================================

import sys
import os
import numpy as np

import torch
import torch_npu

# Reference implementation from origin
ORIGIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), 'origin')
sys.path.insert(0, ORIGIN_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden import bf16_to_fp32, fp32_to_bf16, fp32_to_bf16_rne, compute_golden

# Load the custom op
SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'libmhc_post_ops.so')
if not os.path.exists(SO_PATH):
    SO_PATH_ALT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'libmhc_post_ops.so')
    SO_PATH = os.path.abspath(SO_PATH_ALT)

if not os.path.exists(SO_PATH):
    print(f"ERROR: libmhc_post_ops.so not found. Build first with cmake/make.")
    sys.exit(1)

torch.ops.load_library(SO_PATH)


def reference_mhc_post(x_bf16, residual_bf16, post_layer_mix, comb_res_mix):
    """PyTorch reference matching mhc_post.py Model.forward"""
    # Inputs: x_bf16=(n0,n1,h), residual_bf16=(n0,n1,M,h), pm=(n0,n1,M,1), cmb=(n0,n1,M,M)
    x_f = x_bf16.float()
    res_f = residual_bf16.float()
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, res_f)
    out = (x_f.unsqueeze(-2) * post_layer_mix + term2).bfloat16()
    return out


def run_test(name, x, residual, pm, cmb, atol=0.07, rtol=1e-2):
    """Run one test case. x, residual are bfloat16; pm, cmb are float32."""
    # Move to NPU
    x_npu    = x.npu()
    res_npu  = residual.npu()
    pm_npu   = pm.npu()
    cmb_npu  = cmb.npu()

    # Custom op
    y_npu = torch.ops.npu.mhc_post(x_npu, res_npu, pm_npu, cmb_npu)

    # Reference: use numpy compute_golden which matches kernel's sequential Muls+Add
    # (torch.einsum accumulates in a different order, causing 1-ULP bf16 differences)
    x_np   = x.float().numpy()
    res_np = residual.float().numpy()
    pm_np  = pm.numpy()
    cmb_np = cmb.numpy()
    golden_np = compute_golden(x_np, res_np, pm_np, cmb_np)
    y_ref = torch.from_numpy(golden_np)

    # Compare on CPU
    y_cpu = y_npu.cpu().float()

    diff = torch.abs(y_cpu - y_ref)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # MERE/MARE
    denom = torch.abs(y_ref) + 1e-7
    rel = diff / denom
    mere = rel.mean().item()
    valid_mask = torch.abs(y_ref) > (1e-5 + 1e-7)
    if valid_mask.any():
        mare = rel[valid_mask].max().item()
    else:
        mare = 0.0

    mere_thr = 2.0 ** -7   # ~7.81e-3
    mare_thr = 10.0 * mere_thr

    passed = (mere < mere_thr) and (mare < mare_thr)
    print(f"  [{name}] max_diff={max_diff:.4e} MERE={mere:.4e}({'PASS' if mere < mere_thr else 'FAIL'}) "
          f"MARE={mare:.4e}({'PASS' if mare < mare_thr else 'FAIL'}) "
          f"=> {'PASSED' if passed else 'FAILED'}")
    return name, passed, max_diff


def main():
    print("=== MHC Post PyTorch Extension Test ===")
    print(f"Loaded: {SO_PATH}")

    results = []

    # TC-P1: Standard shape (2, 4096, 4, 1280)
    torch.manual_seed(42)
    n0, n1, M, h = 2, 4096, 4, 1280
    x  = torch.randn(n0, n1, h, dtype=torch.bfloat16)
    rs = torch.randn(n0, n1, M, h, dtype=torch.bfloat16)
    pm = torch.randn(n0, n1, M, 1, dtype=torch.float32)
    cm = torch.randn(n0, n1, M, M, dtype=torch.float32)
    results.append(run_test("TC-P1 standard (2,4096,4,1280)", x, rs, pm, cm))

    # TC-P2: All-zero input
    n0, n1, M, h = 2, 4096, 4, 1280
    x  = torch.zeros(n0, n1, h, dtype=torch.bfloat16)
    rs = torch.zeros(n0, n1, M, h, dtype=torch.bfloat16)
    pm = torch.zeros(n0, n1, M, 1, dtype=torch.float32)
    cm = torch.zeros(n0, n1, M, M, dtype=torch.float32)
    results.append(run_test("TC-P2 all-zero", x, rs, pm, cm))

    # TC-P3: Small shape (1, 1, 4, 64)
    torch.manual_seed(100)
    n0, n1, M, h = 1, 1, 4, 64
    x  = torch.randn(n0, n1, h, dtype=torch.bfloat16)
    rs = torch.randn(n0, n1, M, h, dtype=torch.bfloat16)
    pm = torch.randn(n0, n1, M, 1, dtype=torch.float32)
    cm = torch.randn(n0, n1, M, M, dtype=torch.float32)
    results.append(run_test("TC-P3 small (1,1,4,64)", x, rs, pm, cm))

    # TC-P4: n0=1 shape
    torch.manual_seed(200)
    n0, n1, M, h = 1, 4096, 4, 1280
    x  = torch.randn(n0, n1, h, dtype=torch.bfloat16)
    rs = torch.randn(n0, n1, M, h, dtype=torch.bfloat16)
    pm = torch.randn(n0, n1, M, 1, dtype=torch.float32)
    cm = torch.randn(n0, n1, M, M, dtype=torch.float32)
    results.append(run_test("TC-P4 n0=1 (1,4096,4,1280)", x, rs, pm, cm))

    # Summary
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print()
    print(f"{'='*50}")
    print(f"PyTorch test results: {passed}/{total} passed, {failed} failed")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
