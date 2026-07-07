# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# PyTorch pathway comprehensive test for head_compute_mix_bwd
# ============================================================================

import sys
import os
import numpy as np

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libhead_compute_mix_bwd_ops.so"
OP_NAME = "head_compute_mix_bwd"
DTYPE = torch.float32
RTOL = 1e-4
ATOL = 1e-6


def run_test(name, input_mix, mhc_scale, mhc_base, grad_out):
    """Run a single test case. Inputs are CPU tensors."""
    op_fn = getattr(torch.ops.npu, OP_NAME)

    # Move inputs to NPU for kernel execution
    im_npu = input_mix.npu()
    ms_npu = mhc_scale.npu()
    mb_npu = mhc_base.npu()
    go_npu = grad_out.npu()

    y1_npu, y2_npu, y3_npu = op_fn(im_npu, ms_npu, mb_npu, go_npu)

    # Compute golden on CPU using numpy (for reliable reference)
    g1, g2, g3 = compute_golden(
        input_mix.numpy(),
        mhc_scale.numpy(),
        mhc_base.numpy(),
        grad_out.numpy())

    # Compare on CPU
    y1_cpu = y1_npu.cpu()
    y2_cpu = y2_npu.cpu()
    y3_cpu = y3_npu.cpu()
    g1_t = torch.from_numpy(g1)
    g2_t = torch.from_numpy(g2)
    g3_t = torch.from_numpy(g3)

    d1 = torch.max(torch.abs(y1_cpu - g1_t)).item()
    d2 = torch.max(torch.abs(y2_cpu - g2_t)).item()
    d3 = torch.max(torch.abs(y3_cpu - g3_t)).item()
    max_diff = max(d1, d2, d3)

    p1 = torch.allclose(y1_cpu, g1_t, atol=ATOL, rtol=RTOL)
    p2 = torch.allclose(y2_cpu, g2_t, atol=ATOL, rtol=RTOL)
    p3 = torch.allclose(y3_cpu, g3_t, atol=ATOL, rtol=RTOL)
    passed = p1 and p2 and p3

    return name, passed, max_diff, (d1, d2, d3)


def main():
    so_path = os.path.join(".", SO_NAME)
    if not os.path.exists(so_path):
        so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run cmake and make first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []
    C = 4  # mhc_mult is fixed

    # ========================================================================
    # FT: Functional Tests
    # ========================================================================
    B, S = 2, 1024
    im = torch.randn(B, S, C, dtype=DTYPE)
    ms = torch.randn(1, dtype=DTYPE)
    mb = torch.randn(C, dtype=DTYPE)
    go = torch.randn(B, S, C, dtype=DTYPE)
    results.append(run_test("FT-01: Standard (2x1024x4)", im, ms, mb, go))

    B2, S2 = 1, 1
    im2 = torch.randn(B2, S2, C, dtype=DTYPE)
    ms2 = torch.randn(1, dtype=DTYPE)
    mb2 = torch.randn(C, dtype=DTYPE)
    go2 = torch.randn(B2, S2, C, dtype=DTYPE)
    results.append(run_test("FT-02: Minimum (1x1x4)", im2, ms2, mb2, go2))

    B3, S3 = 4, 512
    im3 = torch.randn(B3, S3, C, dtype=DTYPE)
    ms3 = torch.randn(1, dtype=DTYPE)
    mb3 = torch.randn(C, dtype=DTYPE)
    go3 = torch.randn(B3, S3, C, dtype=DTYPE)
    results.append(run_test("FT-03: Asymmetric (4x512x4)", im3, ms3, mb3, go3))

    B4, S4 = 2, 4096
    im4 = torch.randn(B4, S4, C, dtype=DTYPE)
    ms4 = torch.randn(1, dtype=DTYPE)
    mb4 = torch.randn(C, dtype=DTYPE)
    go4 = torch.randn(B4, S4, C, dtype=DTYPE)
    results.append(run_test("FT-04: Large n1 (2x4096x4)", im4, ms4, mb4, go4))

    B5, S5 = 3, 2048
    im5 = torch.randn(B5, S5, C, dtype=DTYPE)
    ms5 = torch.randn(1, dtype=DTYPE)
    mb5 = torch.randn(C, dtype=DTYPE)
    go5 = torch.randn(B5, S5, C, dtype=DTYPE)
    results.append(run_test("FT-05: Random (3x2048x4)", im5, ms5, mb5, go5))

    # ========================================================================
    # BT: Boundary Tests
    # ========================================================================
    im_z = torch.zeros(B, S, C, dtype=DTYPE)
    ms_z = torch.randn(1, dtype=DTYPE)
    mb_z = torch.zeros(C, dtype=DTYPE)
    go_z = torch.randn(B, S, C, dtype=DTYPE)
    results.append(run_test("BT-01: Zeros input_mix", im_z, ms_z, mb_z, go_z))

    im_large = torch.full((B, S, C), 100.0, dtype=DTYPE)
    results.append(run_test("BT-02: input_mix=+100", im_large, ms, mb, go))

    im_neg = torch.full((B, S, C), -100.0, dtype=DTYPE)
    results.append(run_test("BT-03: input_mix=-100", im_neg, ms, mb, go))

    ms_zero = torch.zeros(1, dtype=DTYPE)
    results.append(run_test("BT-04: mhc_scale=0", im, ms_zero, mb, go))

    mb_diverse = torch.tensor([-10.0, 0.0, 5.0, 20.0], dtype=DTYPE)
    results.append(run_test("BT-05: diverse mhc_base", im, ms, mb_diverse, go))

    # ========================================================================
    # Level 0: Tiny data (8-16 elements)
    # ========================================================================
    B_t, S_t = 1, 2
    im_t = torch.randn(B_t, S_t, C, dtype=DTYPE)
    ms_t = torch.randn(1, dtype=DTYPE)
    mb_t = torch.randn(C, dtype=DTYPE)
    go_t = torch.randn(B_t, S_t, C, dtype=DTYPE)
    results.append(run_test("L0: Tiny (1x2x4=8 elems)", im_t, ms_t, mb_t, go_t))

    B_t2, S_t2 = 1, 4
    im_t2 = torch.randn(B_t2, S_t2, C, dtype=DTYPE)
    ms_t2 = torch.randn(1, dtype=DTYPE)
    mb_t2 = torch.randn(C, dtype=DTYPE)
    go_t2 = torch.randn(B_t2, S_t2, C, dtype=DTYPE)
    results.append(run_test("L0: Tiny (1x4x4=16 elems)", im_t2, ms_t2, mb_t2, go_t2))

    # Summary
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*70}")
    print(f"  PyTorch Comprehensive Test Results ({OP_NAME})")
    print(f"  Precision: rtol={RTOL}, atol={ATOL}")
    print(f"{'='*70}")
    for name, ok, diff, details in results:
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}: {status}  (Max diff={diff:.6e}, parts={details[0]:.3e},{details[1]:.3e},{details[2]:.3e})")
    print(f"{'='*70}")
    print(f"  Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"  Status: {'PASSED' if failed == 0 else 'FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
