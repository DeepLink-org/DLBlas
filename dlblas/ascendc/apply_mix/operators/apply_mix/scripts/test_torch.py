# apply_mix PyTorch pathway test

import sys
import os
import torch
import torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from golden import compute_golden, fp32_to_bf16_uint16, bf16_uint16_to_fp32

SO_NAME = "libapply_mix_ops.so"
OP_NAME = "apply_mix"

MERE_THRESHOLD = 0.00781
MARE_THRESHOLD = 0.0781


def compute_metrics(output_tensor, golden_tensor):
    """Compute MERE/MARE between output and golden (both bf16)."""
    output_fp32 = output_tensor.float()
    golden_fp32 = golden_tensor.float()
    abs_diff = torch.abs(output_fp32 - golden_fp32)
    rel_diff = torch.zeros_like(abs_diff)
    nonzero_mask = torch.abs(golden_fp32) > 1e-30
    rel_diff[nonzero_mask] = abs_diff[nonzero_mask] / torch.abs(golden_fp32[nonzero_mask])

    mere = rel_diff[nonzero_mask].mean().item() if nonzero_mask.any() else 0.0
    mare = rel_diff.max().item()
    return mere, mare, abs_diff.max().item()


def run_test(name, n0, n1, mhc, h):
    """Run a single test case."""
    # Generate data on CPU
    x_fp32 = torch.sigmoid(torch.randn(n0, n1, mhc, h))
    x_bf16 = x_fp32.bfloat16()

    mix_fp32 = torch.nn.functional.softmax(torch.randn(n0, n1, mhc, 1), dim=2)

    # Golden: (x.fp32 * mix).sum(-2).bf16
    golden = (x_bf16.float() * mix_fp32).sum(dim=-2).bfloat16()

    # Move to NPU
    x_npu = x_bf16.npu()
    mix_npu = mix_fp32.npu()

    # Run custom op
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y_npu = op_fn(x_npu, mix_npu)

    y_cpu = y_npu.cpu()
    golden_cpu = golden.cpu()

    expected_shape = (n0, n1, h)
    if tuple(y_cpu.shape) != expected_shape:
        return name, False, float('nan'), float('nan'), float('nan')

    mere, mare, max_abs_diff = compute_metrics(y_cpu, golden_cpu)
    passed = mere < MERE_THRESHOLD and mare < MARE_THRESHOLD
    return name, passed, max_abs_diff, mere, mare


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        # Try current directory (when run from build/)
        so_path = os.path.join(os.getcwd(), SO_NAME)
    if not os.path.exists(so_path):
        # Try relative to script
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run build first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    test_cases = [
        ("TC1 typical", 2, 1024, 4, 1280),
        ("TC2 min", 1, 1, 1, 64),
        ("TC3 medium_mhc", 1, 512, 8, 256),
        ("TC4 large_h", 4, 1, 4, 2048),
        ("TC5 single_batch", 1, 1, 4, 1280),
        ("TC6 tail_non_aligned", 2, 1024, 4, 1300),
        ("TC7 tiny_h", 1, 1, 1, 1),
    ]

    results = []
    for tc_name, n0, n1, mhc, h in test_cases:
        print(f"\n--- {tc_name}: n0={n0}, n1={n1}, mhc={mhc}, h={h} ---")
        try:
            name, passed, max_diff, mere, mare = run_test(tc_name, n0, n1, mhc, h)
            status = "PASSED" if passed else "FAILED"
            print(f"  {status}: MERE={mere:.6f}, MARE={mare:.6f}, max_abs_diff={max_diff:.6e}")
            results.append((name, passed))
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((tc_name, False))

    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'}")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
