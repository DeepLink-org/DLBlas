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
# PyTorch 通路测试脚本 — pre_split_mixes
# ============================================================================

import sys
import os

import numpy as np
import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libpre_split_mixes_ops.so"
OP_NAME = "pre_split_mixes"
DTYPE = torch.float32
ATOL = 1e-5
RTOL = 1e-3

# 测试用例 (与 gen_data.py 一致)
TEST_CASES = {
    "T1": {"batch": 1, "seq_len": 1,    "m": 4},
    "T2": {"batch": 1, "seq_len": 1024, "m": 4},
    "T3": {"batch": 8, "seq_len": 512,  "m": 4},
    "T4": {"batch": 1, "seq_len": 2048, "m": 4},
    "T5": {"batch": 1, "seq_len": 1024, "m": 1},
    "T6": {"batch": 1, "seq_len": 1024, "m": 8},
    "T7": {"batch": 1, "seq_len": 1024, "m": 16},
    "T8": {"batch": 2, "seq_len": 256,  "m": 4},
}

DEFAULT_EPS = 1e-2
DEFAULT_POST_MULT = 2.0


def run_test(name, batch, seq_len, m):
    """运行单个测试用例，返回 (name, passed, info)"""
    M3 = 2 * m + m * m
    np.random.seed(42)

    input_mixes = torch.randn(batch, seq_len, M3, dtype=DTYPE)
    mhc_scale = (torch.randn(3) * 0.1).to(DTYPE)
    mhc_base = (torch.randn(M3) * 0.1).to(DTYPE)

    op_fn = getattr(torch.ops.npu, OP_NAME)
    pre_mix, post_mix, comb_mix = op_fn(
        input_mixes.npu(), mhc_scale.npu(), mhc_base.npu(),
        m, DEFAULT_EPS, DEFAULT_POST_MULT
    )

    # 计算 Golden
    pre_golden, post_golden, comb_golden = compute_golden(
        input_mixes.numpy(), mhc_scale.numpy(), mhc_base.numpy(),
        m, DEFAULT_EPS, DEFAULT_POST_MULT
    )

    pre_golden_t = torch.from_numpy(pre_golden).npu()
    post_golden_t = torch.from_numpy(post_golden).npu()
    comb_golden_t = torch.from_numpy(comb_golden).npu()

    max_diffs = []
    all_passed = True
    for tag, out, gold in [
        ("pre", pre_mix, pre_golden_t),
        ("post", post_mix, post_golden_t),
        ("comb", comb_mix, comb_golden_t),
    ]:
        diff = torch.max(torch.abs(out - gold)).item()
        ok = torch.allclose(out.cpu(), gold.cpu(), atol=ATOL, rtol=RTOL)
        max_diffs.append((tag, diff, ok))
        if not ok:
            all_passed = False

    info = ", ".join(f"{t}={d:.2e}" for t, d, _ in max_diffs)
    return name, all_passed, info


def main():
    # 从 build/ 目录运行时, .so 在当前目录; 从项目根运行时, .so 在 build/ 子目录
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        so_path = SO_NAME  # try current dir
    if not os.path.exists(so_path):
        print(f"ERROR: {SO_NAME} not found in ./build/ or ./. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    case_id = sys.argv[1] if len(sys.argv) > 1 else None

    if case_id and case_id in TEST_CASES:
        cases = {case_id: TEST_CASES[case_id]}
    else:
        cases = TEST_CASES

    results = []
    for cid, tc in cases.items():
        name = f"{cid} (b={tc['batch']},s={tc['seq_len']},m={tc['m']})"
        n, ok, info = run_test(cid, tc["batch"], tc["seq_len"], tc["m"])
        results.append((n, ok, info))

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, info in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} ({info})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
