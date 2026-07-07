# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# PyTorch path test for engram_fused_weight
#
# Tests the TORCH_LIBRARY registered operator via torch.ops.npu.engram_fused_weight
#
# Per DESIGN.md: BF16 inputs → FP32 output
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

# Operator configuration
SO_NAME = "libengram_fused_weight_ops.so"
OP_NAME = "engram_fused_weight"
INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float32
# Per DESIGN.md §8.1: FP32 output precision thresholds
RTOL = 2.0**(-13)       # ≈ 0.000122
ATOL = 2.0**(-13)


def run_test(name, wh_data, we_data):
    """Run single test case, return (name, passed, max_diff, mere)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(wh_data.npu(), we_data.npu())

    # Golden per DESIGN.md §8.3: wh.float() * we.float() in FP32
    golden = compute_golden(wh_data.float(), we_data.float())

    # Compare FP32 to FP32
    y_cpu = y.float().cpu()
    golden_cpu = golden.float().cpu()

    diff = torch.abs(y_cpu - golden_cpu)
    max_diff = diff.max().item()

    # MERE over non-zero golden values
    finite_mask = (torch.abs(golden_cpu) > 1e-10) & torch.isfinite(golden_cpu)
    if finite_mask.sum() > 0:
        mere = (diff[finite_mask] / torch.abs(golden_cpu[finite_mask])).mean().item()
    else:
        mere = 0.0

    passed = torch.allclose(y_cpu, golden_cpu, atol=ATOL, rtol=RTOL)
    return name, passed, max_diff, mere


def main():
    # Load operator library
    script_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(script_dir, "..", "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # TC-01: Standard case (4, 128)
    wh = torch.randn(4, 128, dtype=INPUT_DTYPE)
    we = torch.randn(4, 128, dtype=INPUT_DTYPE)
    results.append(run_test("TC-01 standard (4,128)", wh, we))

    # TC-02: Minimum hc_mult (1, 128)
    wh = torch.randn(1, 128, dtype=INPUT_DTYPE)
    we = torch.randn(1, 128, dtype=INPUT_DTYPE)
    results.append(run_test("TC-02 min_hc_mult (1,128)", wh, we))

    # TC-03: Minimum hidden_size (4, 1)
    wh = torch.randn(4, 1, dtype=INPUT_DTYPE)
    we = torch.randn(4, 1, dtype=INPUT_DTYPE)
    results.append(run_test("TC-03 min_hidden (4,1)", wh, we))

    # TC-04: Single element (1, 1)
    wh = torch.randn(1, 1, dtype=INPUT_DTYPE)
    we = torch.randn(1, 1, dtype=INPUT_DTYPE)
    results.append(run_test("TC-04 single (1,1)", wh, we))

    # TC-05: Larger shape (8, 256)
    wh = torch.randn(8, 256, dtype=INPUT_DTYPE)
    we = torch.randn(8, 256, dtype=INPUT_DTYPE)
    results.append(run_test("TC-05 large (8,256)", wh, we))

    # TC-20: All zeros
    wh = torch.zeros(4, 128, dtype=INPUT_DTYPE)
    we = torch.zeros(4, 128, dtype=INPUT_DTYPE)
    results.append(run_test("TC-20 zeros", wh, we))

    # TC-23: Mixed signs
    wh = torch.randn(4, 128, dtype=INPUT_DTYPE)
    we = -wh
    results.append(run_test("TC-23 pos_neg", wh, we))

    # Summary
    total = len(results)
    passed_count = sum(r[1] for r in results)
    failed = total - passed_count
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME}): {INPUT_DTYPE} → {OUTPUT_DTYPE}")
    print(f"{'='*60}")
    for name, ok, diff, mere in results:
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}: {status} (Max diff={diff:.6e}, MERE={mere:.6e})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed_count}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
