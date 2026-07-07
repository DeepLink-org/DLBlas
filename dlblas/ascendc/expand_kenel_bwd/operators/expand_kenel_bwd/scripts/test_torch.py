# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# PyTorch 通路测试 - Expand Kernel Backward 算子
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libexpand_kenel_bwd_ops.so"
OP_NAME = "expand_kenel_bwd"
DTYPE = torch.float16
ATOL = 1e-3
RTOL = 1e-3


def run_test(name, o_grad):
    """运行单个测试用例，返回 (name, passed, max_diff)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(o_grad.npu())
    golden = compute_golden(o_grad).npu()
    max_diff = torch.max(torch.abs(y.float() - golden.float())).item()
    passed = torch.allclose(y.float().cpu(), golden.float().cpu(),
                            atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    # 从脚本所在目录查找 .so
    script_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(script_dir, "..", "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # T1: 标准 shape (2, 1024, 4, 1280)
    o_grad = torch.randn(2, 1024, 4, 1280, dtype=DTYPE)
    results.append(run_test("T1 standard (2,1024,4,1280)", o_grad))

    # T2: 小 shape (1, 1, 4, 128)
    o_grad = torch.randn(1, 1, 4, 128, dtype=DTYPE)
    results.append(run_test("T2 small (1,1,4,128)", o_grad))

    # T3: 边界值 - 零值输入
    o_grad = torch.zeros(2, 1024, 4, 1280, dtype=DTYPE)
    results.append(run_test("T3 zeros (2,1024,4,1280)", o_grad))

    # T4: 极值输入
    o_grad = torch.ones(2, 256, 4, 128, dtype=DTYPE) * 100.0
    results.append(run_test("T4 large values (2,256,4,128)", o_grad))

    # T5: 正负混合
    o_grad = torch.randn(2, 512, 4, 256, dtype=DTYPE) * 10.0
    results.append(run_test("T5 mixed signs (2,512,4,256)", o_grad))

    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6e})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
