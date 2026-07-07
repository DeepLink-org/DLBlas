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
# PyTorch 通路测试脚本 - Expand 算子
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libexpand_kenel_fwd_ops.so"
OP_NAME = "expand_kenel_fwd"


def run_test(name, x, mhc_mult):
    """运行单个测试用例，返回 (name, passed, max_diff, mismatch_count)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x.npu(), mhc_mult)

    # Golden 计算
    golden = compute_golden(x.cpu(), mhc_mult)
    if hasattr(golden, 'numpy'):
        golden_npu = golden.npu()
    else:
        golden_npu = torch.from_numpy(golden).npu()

    # Bitwise match
    y_cpu = y.cpu()
    golden_cpu = golden_npu.cpu()

    total_elements = y_cpu.numel()
    mismatches = (y_cpu.contiguous().view(-1) != golden_cpu.contiguous().view(-1)).sum().item()

    if mismatches == 0:
        max_diff = 0.0
    else:
        max_diff = torch.max(torch.abs(y_cpu.float() - golden_cpu.float())).item()

    passed = (mismatches == 0)
    return name, passed, max_diff, mismatches, total_elements


def main():
    # 加载算子库
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", SO_NAME)
    if not os.path.exists(so_path):
        # Fallback: look in cwd/build
        so_path = os.path.join(os.getcwd(), "build", SO_NAME)
    if not os.path.exists(so_path):
        # Fallback: look in cwd
        so_path = os.path.join(os.getcwd(), SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {SO_NAME} not found. Searched: {so_path}")
        sys.exit(1)
    torch.ops.load_library(so_path)

    # 测试用例
    results = []

    # T1: 典型用例 (FP16) - B=1, S=1024, H=1280, M=4
    x = torch.randn(1, 1024, 1280, dtype=torch.float16)
    results.append(run_test("T1 typical FP16", x, 4))

    # T2: 最小行数 - B=1, S=1, H=128, M=2
    x = torch.randn(1, 1, 128, dtype=torch.float16)
    results.append(run_test("T2 min rows FP16", x, 2))

    # T3: 多行 - B=4, S=256, H=256, M=2
    x = torch.randn(4, 256, 256, dtype=torch.float16)
    results.append(run_test("T3 multi rows FP16", x, 2))

    # T4: 大M - B=1, S=1, H=1280, M=16
    x = torch.randn(1, 1, 1280, dtype=torch.float16)
    results.append(run_test("T4 large M FP16", x, 16))

    # T5: M=1 退化场景
    x = torch.randn(1, 1, 1280, dtype=torch.float16)
    results.append(run_test("T5 M=1 FP16", x, 1))

    # T6: FP32 精度
    x = torch.randn(1, 1024, 1280, dtype=torch.float32)
    results.append(run_test("T6 FP32", x, 4))

    # T7: 对齐边界 (H=32, 最小对齐倍数)
    x = torch.randn(1, 5, 32, dtype=torch.float16)
    results.append(run_test("T7 aligned H=32", x, 4))

    # T8: 多核场景 - B=10, S=100, H=512, M=8
    x = torch.randn(10, 100, 512, dtype=torch.float16)
    results.append(run_test("T8 multicore", x, 8))

    # T9: 大 H
    x = torch.randn(1, 1, 2048, dtype=torch.float16)
    results.append(run_test("T9 large H", x, 4))

    # T10: BF16
    x = torch.randn(1, 16, 128, dtype=torch.bfloat16)
    results.append(run_test("T10 BF16", x, 4))

    # === 非对齐 H 拒绝测试 (L4 fix) ===
    # 这些测试验证 H 非 16 倍数时 Host 侧正确拒绝

    # T11: H=33 (非对齐) — 应被 TORCH_CHECK 拒绝
    try:
        x = torch.randn(1, 5, 33, dtype=torch.float16)
        op_fn = getattr(torch.ops.npu, OP_NAME)
        y = op_fn(x.npu(), 4)
        results.append(("T11 non-aligned H=33", False, -1.0, -1, 0))
    except RuntimeError:
        results.append(("T11 non-aligned H=33 (rejected as expected)", True, 0.0, 0, 0))

    # T12: H=37 (非对齐) — 应被 TORCH_CHECK 拒绝
    try:
        x = torch.randn(1, 5, 37, dtype=torch.float16)
        op_fn = getattr(torch.ops.npu, OP_NAME)
        y = op_fn(x.npu(), 4)
        results.append(("T12 non-aligned H=37", False, -1.0, -1, 0))
    except RuntimeError:
        results.append(("T12 non-aligned H=37 (rejected as expected)", True, 0.0, 0, 0))

    # T13: 常用对齐值 H=256 额外验证
    x = torch.randn(1, 1, 256, dtype=torch.float16)
    results.append(run_test("T13 aligned H=256", x, 4))

    # T14: 大 batch 多核负载
    x = torch.randn(8, 1024, 1280, dtype=torch.float16)
    results.append(run_test("T14 large batch multicore", x, 4))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, diff, mism, total_elem in results:
        status = "PASSED (bitwise)" if ok else "FAILED"
        print(f"  {name}: {status} (mismatches={mism}/{total_elem}, max_diff={diff})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
