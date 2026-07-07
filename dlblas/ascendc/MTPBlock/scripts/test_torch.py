# ============================================================================
# MTPBlock PyTorch 通路测试脚本 - K4 hc_post
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden_k4_hc_post

SO_NAME = "libmtpblock_ops.so"
OP_NAME = "hc_post"
DTYPE = torch.bfloat16
ATOL = 1e-2
RTOL = 5e-3


def run_test(name, x, residual, post, comb):
    """Run single test case"""
    op_fn = getattr(torch.ops.mtpblock, OP_NAME)
    y = op_fn(x.npu(), residual.npu(), post.npu(), comb.npu())

    golden_np = compute_golden_k4_hc_post(
        x.float().numpy(),
        residual.float().numpy(),
        post.float().numpy(),
        comb.float().numpy()
    )
    golden = torch.from_numpy(golden_np).bfloat16().npu()

    max_diff = torch.max(torch.abs(y.float() - golden.float())).item()
    passed = torch.allclose(y.float().cpu(), golden.float().cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run build first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    b, s, hc, d = 1, 8, 4, 512
    results = []

    # Test 1: Random data
    x0 = torch.randn(b, s, d, dtype=DTYPE)
    residual0 = torch.randn(b, s, hc, d, dtype=DTYPE)
    post0 = torch.randn(b, s, hc, dtype=torch.float32)
    comb0 = torch.randn(b, s, hc, hc, dtype=torch.float32)
    results.append(run_test("T1 random", x0, residual0, post0, comb0))

    # Test 2: Zeros
    x1 = torch.zeros(b, s, d, dtype=DTYPE)
    residual1 = torch.zeros(b, s, hc, d, dtype=DTYPE)
    post1 = torch.zeros(b, s, hc, dtype=torch.float32)
    comb1 = torch.zeros(b, s, hc, hc, dtype=torch.float32)
    results.append(run_test("T2 zeros", x1, residual1, post1, comb1))

    # Test 3: Ones
    x2 = torch.ones(b, s, d, dtype=DTYPE)
    residual2 = torch.ones(b, s, hc, d, dtype=DTYPE)
    post2 = torch.ones(b, s, hc, dtype=torch.float32)
    comb2 = torch.eye(hc, hc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(b, s, -1, -1)
    results.append(run_test("T3 ones+identity", x2, residual2, post2, comb2))

    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6f})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
