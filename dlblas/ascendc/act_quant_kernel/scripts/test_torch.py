# ----------------------------------------------------------------------------
# PyTorch 通路测试 - act_quant_kernel
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libact_quant_kernel_ops.so"
OP_NAME = "act_quant_kernel"
DTYPE = torch.bfloat16
GROUP_SIZE = 128
EPS = 1e-10


def run_test(name, x):
    """Run a single test case."""
    x_npu = x.npu()
    x_q, x_s = torch.ops.npu.act_quant_kernel(x_npu, GROUP_SIZE, EPS, False)

    # Compute golden
    x_np = x.float().cpu().numpy().reshape(-1)
    q_golden, s_golden = compute_golden(x_np, GROUP_SIZE, EPS, False)

    # Compare x_q (uint8 exact match)
    q_out = x_q.cpu().numpy().reshape(-1).astype(int)
    q_g = q_golden.astype(int)
    q_mismatch = int((q_out != q_g).sum())
    q_passed = q_mismatch == 0
    # Check if mismatches are > 1 ULP
    if not q_passed:
        q_diff = np.abs(q_out.astype(np.int32) - q_g.astype(np.int32))
        q_big = int((q_diff > 1).sum())
        q_passed = q_big == 0
        if q_passed:
            print(f"    Note: {q_mismatch} elements with 1-ULP diff (acceptable)")

    # Compare x_s (fp32 approximate match)
    s_out = x_s.cpu().numpy().reshape(-1)
    s_g = s_golden.reshape(-1)
    s_max_diff = float(abs(s_out - s_g).max())
    s_passed = bool(s_max_diff < 1e-4)

    passed = q_passed and s_passed
    return name, passed, q_mismatch, s_max_diff


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    so_candidates = [
        os.path.join(os.getcwd(), SO_NAME),
        os.path.join(script_dir, "..", "build", SO_NAME),
    ]
    so_path = None
    for p in so_candidates:
        if os.path.exists(p):
            so_path = p
            break
    if so_path is None:
        print(f"ERROR: {SO_NAME} not found. Searched: {so_candidates}")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # T1: Level 0 - minimal shape
    x1 = torch.randn(1, 128, dtype=DTYPE)
    results.append(run_test("T1 [1,128] gs=128", x1))

    # T2: Level 0 - small multi-group
    x2 = torch.randn(8, 128, dtype=DTYPE)
    results.append(run_test("T2 [8,128] gs=128", x2))

    # Summary
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, q_mis, s_diff in results:
        status = "PASSED" if ok else "FAILED"
        print(f"  {name}: {status} (q_mismatch={q_mis}, s_max_diff={s_diff:.6f})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
