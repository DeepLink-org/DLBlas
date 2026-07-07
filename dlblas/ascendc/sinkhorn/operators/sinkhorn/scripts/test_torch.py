# Sinkhorn Normalize - PyTorch 通路测试

import sys
import os

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libsinkhorn_ops.so"
OP_NAME = "sinkhorn_normalize"
DTYPE = torch.float32
ATOL = 1e-5
RTOL = 1e-5


def run_test(name, x, mhc=4, repeat=10, eps=1e-6):
    """运行单个测试用例"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    x_npu = x.npu()
    y = op_fn(x_npu)
    golden = torch.from_numpy(compute_golden(x.numpy(), mhc=mhc, repeat=repeat, eps=eps)).npu()
    max_diff = torch.max(torch.abs(y - golden)).item()
    passed = torch.allclose(y.cpu(), golden.cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    # Find .so: try multiple locations
    so_path = None
    for p in [SO_NAME, os.path.join("build", SO_NAME), os.path.join("..", "build", SO_NAME)]:
        if os.path.exists(p):
            so_path = p
            break
    if so_path is None:
        print(f"ERROR: {SO_NAME} not found. Run cmake && make first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # TC001: 单矩阵基本功能
    x = torch.randn(1, 1, 4, 4, dtype=DTYPE)
    results.append(run_test("TC001 single_matrix", x))

    # TC002: 全量 batch
    x = torch.randn(1, 1024, 4, 4, dtype=DTYPE)
    results.append(run_test("TC002 full_batch", x))

    # TC003: 随机矩阵 (additional random test)
    torch.manual_seed(123)
    x = torch.randn(1, 8, 4, 4, dtype=DTYPE)
    results.append(run_test("TC003 random_8batch", x))

    # TC004: 小 batch
    x = torch.randn(1, 4, 4, 4, dtype=DTYPE)
    results.append(run_test("TC004 small_batch", x))

    # TC006: 全零输入
    x = torch.zeros(1, 1, 4, 4, dtype=DTYPE)
    results.append(run_test("TC006 zeros", x))

    # TC007: 极大正值
    torch.manual_seed(42)
    x = torch.ones(1, 1, 4, 4, dtype=DTYPE) * 1000.0
    results.append(run_test("TC007 large_positive", x))

    # TC008: 极负值
    x = torch.ones(1, 1, 4, 4, dtype=DTYPE) * (-1000.0)
    results.append(run_test("TC008 large_negative", x))

    # TC009: batch 不可整除
    x = torch.randn(1, 63, 4, 4, dtype=DTYPE)
    results.append(run_test("TC009 non_divisible_batch", x))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
