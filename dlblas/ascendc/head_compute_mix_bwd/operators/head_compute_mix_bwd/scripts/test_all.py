# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Comprehensive test runner for head_compute_mix_bwd (Direct Invoke path)
#
# Usage:
#   python3 test_all.py                  # Run all tests
# ============================================================================

import numpy as np
import os
import subprocess
import sys

dtype = np.float32
rtol = 1e-4
atol = 1e-6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "..", "build")
BINARY = os.path.join(BUILD_DIR, "head_compute_mix_bwd")

from golden import compute_golden


def run_one_test(name, n0, n1, mhc_mult, override_data=None):
    """Run a single test case: gen data, run kernel, verify.

    Args:
        name: test case name
        n0, n1, mhc_mult: shape params
        override_data: optional dict with keys 'input_mix','mhc_scale','mhc_base','grad_out'
                       to override the random data
    """
    os.makedirs(os.path.join(BUILD_DIR, "input"), exist_ok=True)
    os.makedirs(os.path.join(BUILD_DIR, "output"), exist_ok=True)

    # Generate data
    if override_data is not None:
        input_mix = override_data['input_mix']
        mhc_scale = override_data['mhc_scale']
        mhc_base = override_data['mhc_base']
        grad_out = override_data['grad_out']
    else:
        np.random.seed(hash(name) & 0xFFFFFFFF)
        input_mix = np.random.randn(n0, n1, mhc_mult).astype(dtype)
        mhc_scale = np.random.randn(1).astype(dtype)
        mhc_base = np.random.randn(mhc_mult).astype(dtype)
        grad_out = np.random.randn(n0, n1, mhc_mult).astype(dtype)

    # Write inputs to binary files
    input_mix.tofile(os.path.join(BUILD_DIR, "input", "input_input_mix.bin"))
    mhc_scale.tofile(os.path.join(BUILD_DIR, "input", "input_mhc_scale.bin"))
    mhc_base_8 = np.concatenate([mhc_base, mhc_base]).astype(dtype)
    mhc_base_8.tofile(os.path.join(BUILD_DIR, "input", "input_mhc_base.bin"))
    grad_out.tofile(os.path.join(BUILD_DIR, "input", "input_grad_out.bin"))

    # Compute golden
    g1, g2, g3 = compute_golden(input_mix, mhc_scale, mhc_base, grad_out)
    g1.tofile(os.path.join(BUILD_DIR, "output", "golden_grad_input_mix.bin"))
    g2.tofile(os.path.join(BUILD_DIR, "output", "golden_grad_mhc_scale.bin"))
    g3.tofile(os.path.join(BUILD_DIR, "output", "golden_grad_mhc_base.bin"))

    # Run kernel
    rm_files = [
        os.path.join(BUILD_DIR, "output", "output_grad_input_mix.bin"),
        os.path.join(BUILD_DIR, "output", "output_grad_mhc_scale.bin"),
        os.path.join(BUILD_DIR, "output", "output_grad_mhc_base.bin"),
    ]
    for f in rm_files:
        if os.path.exists(f):
            os.remove(f)

    result = subprocess.run(
        [BINARY, str(n0), str(n1), str(mhc_mult)],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        return name, False, f"Kernel exit code {result.returncode}\nSTDERR: {result.stderr[-500:]}"

    # Verify each output
    outputs = [
        ("grad_input_mix", "output_grad_input_mix.bin", "golden_grad_input_mix.bin"),
        ("grad_mhc_scale", "output_grad_mhc_scale.bin", "golden_grad_mhc_scale.bin"),
        ("grad_mhc_base", "output_grad_mhc_base.bin", "golden_grad_mhc_base.bin"),
    ]

    diffs = []
    for oname, ofile, gfile in outputs:
        opath = os.path.join(BUILD_DIR, "output", ofile)
        gpath = os.path.join(BUILD_DIR, "output", gfile)
        if not os.path.exists(opath):
            return name, False, f"Output file {ofile} not found"
        out = np.fromfile(opath, dtype=dtype)
        gold = np.fromfile(gpath, dtype=dtype)
        if out.shape != gold.shape:
            return name, False, f"Shape mismatch for {oname}: {out.shape} vs {gold.shape}"
        diff = np.max(np.abs(out - gold))
        diffs.append((oname, diff))
        if not np.allclose(out, gold, rtol=rtol, atol=atol):
            return name, False, f"{oname} failed: max_diff={diff:.6e}"

    max_diff_name, max_diff_val = max(diffs, key=lambda x: x[1])
    return name, True, f"All outputs pass (max_diff={max_diff_val:.6e} in {max_diff_name})"


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary {BINARY} not found. Build first.")
        return 1

    tests = []
    results = []

    # ========================================================================
    # FT: Functional Tests
    # ========================================================================
    tests.append(("FT-01: Standard (2x1024x4)", 2, 1024, 4, None))
    tests.append(("FT-02: Minimum (1x1x4)", 1, 1, 4, None))
    tests.append(("FT-03: Asymmetric (4x512x4)", 4, 512, 4, None))
    tests.append(("FT-04: Large n1 (2x4096x4)", 2, 4096, 4, None))
    tests.append(("FT-05: Random (3x2048x4)", 3, 2048, 4, None))

    # ========================================================================
    # BT: Boundary Tests
    # ========================================================================
    # BT-01: Zeros input
    B, S, C = 2, 1024, 4
    im = np.zeros((B, S, C), dtype=dtype)
    ms = np.random.randn(1).astype(dtype)
    mb = np.zeros(C, dtype=dtype)
    go = np.random.randn(B, S, C).astype(dtype)
    tests.append(("BT-01: Zeros input_mix", B, S, C,
                  {'input_mix': im, 'mhc_scale': ms, 'mhc_base': mb, 'grad_out': go}))

    # BT-02: Large positive input_mix (saturates sigmoid)
    im_large = np.full((B, S, C), 100.0, dtype=dtype)
    tests.append(("BT-02: input_mix=+100", B, S, C,
                  {'input_mix': im_large, 'mhc_scale': ms, 'mhc_base': mb, 'grad_out': go}))

    # BT-03: Large negative input_mix
    im_neg = np.full((B, S, C), -100.0, dtype=dtype)
    tests.append(("BT-03: input_mix=-100", B, S, C,
                  {'input_mix': im_neg, 'mhc_scale': ms, 'mhc_base': mb, 'grad_out': go}))

    # BT-04: Zero mhc_scale
    ms_zero = np.zeros(1, dtype=dtype)
    im_rand = np.random.randn(B, S, C).astype(dtype)
    mb_rand = np.random.randn(C).astype(dtype)
    go_rand = np.random.randn(B, S, C).astype(dtype)
    tests.append(("BT-04: mhc_scale=0", B, S, C,
                  {'input_mix': im_rand, 'mhc_scale': ms_zero, 'mhc_base': mb_rand, 'grad_out': go_rand}))

    # BT-05: Large channel bias differences
    mb_large = np.array([-10.0, 0.0, 5.0, 20.0], dtype=dtype)
    tests.append(("BT-05: diverse mhc_base", B, S, C,
                  {'input_mix': im_rand, 'mhc_scale': ms, 'mhc_base': mb_large, 'grad_out': go_rand}))

    # ========================================================================
    # MC: Multi-Core Tests (single core forced manually is tricky, skip for direct invoke)
    # ========================================================================
    # MC cases are implicitly tested by different shapes that trigger different core counts:
    # FT-02 (1x1x4) → core_num=1
    # FT-01 (2x1024x4) → core_num=8
    # FT-03 (4x512x4) → core_num=8
    tests.append(("MC-implicit: Single core (FT-02)", 1, 1, 4, None))
    tests.append(("MC-implicit: 8 cores (FT-01)", 2, 1024, 4, None))

    # ========================================================================
    # Run all tests
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  Comprehensive Test Suite: head_compute_mix_bwd")
    print(f"  Precision: rtol={rtol}, atol={atol}")
    print(f"  Total tests: {len(tests)}")
    print(f"{'='*70}\n")

    passed = 0
    failed = 0

    for name, n0, n1, mhc_mult, override in tests:
        sys.stdout.write(f"  [{name}] ... ")
        sys.stdout.flush()
        tname, ok, msg = run_one_test(name, n0, n1, mhc_mult, override)
        if ok:
            print(f"PASSED  ({msg})")
            passed += 1
        else:
            print(f"FAILED  ({msg})")
            failed += 1
        results.append((tname, ok, msg))

    # Summary
    print(f"\n{'='*70}")
    print(f"  Results: {passed}/{len(tests)} passed")
    if failed > 0:
        print(f"  Failed tests:")
        for tname, ok, msg in results:
            if not ok:
                print(f"    - {tname}: {msg}")
    print(f"  Status: {'PASSED' if failed == 0 else 'FAILED'}")
    print(f"{'='*70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
