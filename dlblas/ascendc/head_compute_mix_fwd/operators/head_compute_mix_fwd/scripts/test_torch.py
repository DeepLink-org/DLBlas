# ============================================================================
# PyTorch pathway test for head_compute_mix_fwd
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "libhead_compute_mix_fwd_ops.so"
OP_NAME = "head_compute_mix_fwd"
DTYPE = torch.float16
ATOL = 1e-3
RTOL = 1e-2


def run_test(name, input_mix, mhc_scale, mhc_base, mhc_pre_eps):
    """Run a single test case, returns (name, passed, max_diff)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
    golden = compute_golden(input_mix, mhc_scale, mhc_base, mhc_pre_eps).npu()
    max_diff = torch.max(torch.abs(y.float() - golden.float())).item()
    passed = torch.allclose(y.float().cpu(), golden.float().cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(script_dir, "..", "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # P1: Small shape (8 elements)
    x = torch.randn(2, 1, 4, dtype=DTYPE)
    s = torch.randn(1, dtype=DTYPE)
    b = torch.randn(4, dtype=DTYPE)
    results.append(run_test("P1 small_8", x.npu(), s.npu(), b.npu(), 0.01))

    # P2: 1K elements
    x = torch.randn(1, 256, 4, dtype=DTYPE)
    s = torch.randn(1, dtype=DTYPE)
    b = torch.randn(4, dtype=DTYPE)
    results.append(run_test("P2 1K", x.npu(), s.npu(), b.npu(), 0.01))

    # P3: Default shape (1M elements)
    x = torch.randn(16, 16384, 4, dtype=DTYPE)
    s = torch.randn(1, dtype=DTYPE)
    b = torch.randn(4, dtype=DTYPE)
    results.append(run_test("P3 default_1M", x.npu(), s.npu(), b.npu(), 0.01))

    # P4: Zero input
    x = torch.zeros(8, 16, 4, dtype=DTYPE)
    s = torch.ones(1, dtype=DTYPE)
    b = torch.zeros(4, dtype=DTYPE)
    results.append(run_test("P4 zeros", x.npu(), s.npu(), b.npu(), 0.01))

    # P5: Extreme values
    x = torch.tensor([[[10.0, 5.0, -5.0, -10.0]]], dtype=DTYPE)
    s = torch.tensor([1.0], dtype=DTYPE)
    b = torch.zeros(4, dtype=DTYPE)
    results.append(run_test("P5 extreme", x.npu(), s.npu(), b.npu(), 0.0))

    # P6: Non-aligned shape
    x = torch.randn(3, 100, 4, dtype=DTYPE)
    s = torch.randn(1, dtype=DTYPE)
    b = torch.randn(4, dtype=DTYPE)
    results.append(run_test("P6 non_aligned", x.npu(), s.npu(), b.npu(), 0.01))

    # P7: Large negative values
    x = torch.ones(1, 64, 4, dtype=DTYPE) * (-5.0)
    s = torch.ones(1, dtype=DTYPE)
    b = torch.zeros(4, dtype=DTYPE)
    results.append(run_test("P7 large_neg", x.npu(), s.npu(), b.npu(), 0.01))

    # P8: Asymmetric base
    x = torch.ones(2, 256, 4, dtype=DTYPE)
    s = torch.tensor([2.0], dtype=DTYPE)
    b = torch.tensor([0.1, 1.0, -0.5, -2.0], dtype=DTYPE)
    results.append(run_test("P8 asymmetric", x.npu(), s.npu(), b.npu(), 0.01))

    # Summary
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
