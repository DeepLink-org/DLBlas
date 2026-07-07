# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 测试数据生成脚本 - MHC Post（支持多 shape 测试用例 TC-01 ~ TC-11）
# dtype: bfloat16 I/O, fp32 coefficients
# ============================================================================

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import (compute_golden, generate_test_data, get_test_cases,
                    fp32_to_bf16, write_bf16_bin, bf16_to_fp32, fp32_to_bf16_rne)

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def gen_one(tag, n0, n1, h, mhc_mult, seed, desc):
    """Generate input/output binary files for one test case (bf16 format)."""
    x, residual, post_layer_mix, comb_res_mix = generate_test_data(
        tag, n0, n1, h, mhc_mult, seed)

    # Save inputs as bf16 binary (x and residual), fp32 for coefficients
    write_bf16_bin(x, "input/input_x.bin")
    write_bf16_bin(residual, "input/input_residual.bin")
    post_layer_mix.tofile("input/input_post_layer_mix.bin")
    comb_res_mix.tofile("input/input_comb_res_mix.bin")

    # Compute golden at bf16 precision
    golden = compute_golden(x, residual, post_layer_mix, comb_res_mix)
    # Write golden as bf16 binary (exact match to kernel's CAST_ROUND output)
    golden_bf16 = fp32_to_bf16_rne(golden)
    golden_bf16.tofile("output/golden.bin")

    print(f"[{tag}] {desc}")
    print(f"  shape: n0={n0}, n1={n1}, h={h}, mhc_mult={mhc_mult}")
    print(f"  x: {x.shape} (bf16), residual: {residual.shape} (bf16)")
    print(f"  golden: {golden.shape} (bf16 precision)")
    return tag


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        # Generate all test cases (for batch testing)
        cases = get_test_cases()
        for tag, n0, n1, h, mhc_mult, seed, desc in cases:
            gen_one(tag, n0, n1, h, mhc_mult, seed, desc)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--list":
        # List available test cases
        cases = get_test_cases()
        for tag, n0, n1, h, mhc_mult, seed, desc in cases:
            print(f"{tag}: n0={n0}, n1={n1}, h={h}, mhc={mhc_mult}, seed={seed}  # {desc}")
    elif len(sys.argv) >= 3:
        # Generate specific test case by tag: --test TC-01
        tag = sys.argv[2] if sys.argv[1] == "--test" else sys.argv[1]
        cases = dict((c[0], c) for c in get_test_cases())
        if tag not in cases:
            print(f"Unknown test case: {tag}")
            print(f"Available: {list(cases.keys())}")
            sys.exit(1)
        _, n0, n1, h, mhc_mult, seed, desc = cases[tag]
        gen_one(tag, n0, n1, h, mhc_mult, seed, desc)
    else:
        # Default: standard TC-01
        gen_one("TC-01", 2, 4096, 1280, 4, 42, "Standard shape")
