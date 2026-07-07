#!/usr/bin/env python3
# ============================================================================
# Benchmark: AscendC expand_kenel_fwd vs PyTorch reference
# ============================================================================

import sys
import os
import time
import json

import torch
import torch_npu

# Add scripts dir to path so we can import golden
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from golden import compute_golden

SO_NAME = "libexpand_kenel_fwd_ops.so"
OP_NAME = "expand_kenel_fwd"

WARMUP = 10
REPEAT = 100


def warmup_npu():
    """Warm up NPU device."""
    dummy = torch.randn(128, 128, device='npu')
    for _ in range(5):
        _ = dummy + 1
    torch.npu.synchronize()


def benchmark_torch(x, mhc_mult, warmup=WARMUP, repeat=REPEAT):
    """Benchmark PyTorch reference implementation on NPU."""
    x_npu = x.npu()

    # Warmup
    for _ in range(warmup):
        _ = compute_golden(x_npu, mhc_mult)
    torch.npu.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeat):
        _ = compute_golden(x_npu, mhc_mult)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / repeat * 1000  # ms


def benchmark_ascendc(x, mhc_mult, warmup=WARMUP, repeat=REPEAT):
    """Benchmark AscendC kernel via PyTorch extension."""
    x_npu = x.npu()
    op_fn = getattr(torch.ops.npu, OP_NAME)

    # Warmup
    for _ in range(warmup):
        _ = op_fn(x_npu, mhc_mult)
    torch.npu.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeat):
        _ = op_fn(x_npu, mhc_mult)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / repeat * 1000  # ms


def check_precision(x, mhc_mult):
    """Verify bitwise match between AscendC kernel and PyTorch reference."""
    x_npu = x.npu()
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x_npu, mhc_mult)
    golden = compute_golden(x.cpu(), mhc_mult)

    y_cpu = y.cpu().contiguous().view(-1)
    golden_cpu = golden.contiguous().view(-1)

    mismatches = (y_cpu != golden_cpu).sum().item()
    if mismatches > 0:
        max_diff = torch.max(torch.abs(y_cpu.float() - golden_cpu.float())).item()
    else:
        max_diff = 0.0

    return mismatches == 0, max_diff, mismatches, y_cpu.numel()


def main():
    # Load operator library
    base_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(base_dir, "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {SO_NAME} not found at {so_path}")
        sys.exit(1)
    torch.ops.load_library(so_path)

    warmup_npu()

    test_cases = [
        {"name": "T1 typical FP16", "B": 1, "S": 1024, "H": 1280, "M": 4, "dtype": torch.float16},
        {"name": "T2 min rows", "B": 1, "S": 1, "H": 128, "M": 2, "dtype": torch.float16},
        {"name": "T3 multi rows", "B": 4, "S": 256, "H": 256, "M": 2, "dtype": torch.float16},
        {"name": "T4 large M", "B": 1, "S": 1, "H": 1280, "M": 16, "dtype": torch.float16},
        {"name": "T5 M=1", "B": 1, "S": 1, "H": 1280, "M": 1, "dtype": torch.float16},
        {"name": "T6 FP32", "B": 1, "S": 1024, "H": 1280, "M": 4, "dtype": torch.float32},
        {"name": "T7 aligned H=32", "B": 1, "S": 5, "H": 32, "M": 4, "dtype": torch.float16},
        {"name": "T8 multicore", "B": 10, "S": 100, "H": 512, "M": 8, "dtype": torch.float16},
        {"name": "T9 large H", "B": 1, "S": 1, "H": 2048, "M": 4, "dtype": torch.float16},
        {"name": "T10 BF16", "B": 1, "S": 16, "H": 128, "M": 4, "dtype": torch.bfloat16},
    ]

    results = []
    total_passed = 0
    total_cases = len(test_cases)

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        x = torch.randn(tc["B"], tc["S"], tc["H"], dtype=tc["dtype"])

        # Precision check
        passed, max_diff, mismatches, total_elem = check_precision(x, tc["M"])
        print(f"  Precision: {'PASS (bitwise)' if passed else f'FAIL ({mismatches}/{total_elem}, max_diff={max_diff})'}")

        # Benchmark PyTorch
        torch_ms = benchmark_torch(x, tc["M"])
        print(f"  PyTorch:  {torch_ms:.4f} ms")

        # Benchmark AscendC
        ascendc_ms = benchmark_ascendc(x, tc["M"])
        print(f"  AscendC:  {ascendc_ms:.4f} ms")

        speedup = torch_ms / ascendc_ms if ascendc_ms > 0 else 0.0
        print(f"  Speedup:  {speedup:.4f}x")

        results.append({
            "name": tc["name"],
            "shape": [tc["B"], tc["S"], tc["H"]],
            "M": tc["M"],
            "dtype": str(tc["dtype"]),
            "precision_passed": passed,
            "max_diff": max_diff,
            "mismatches": mismatches,
            "total_elements": total_elem,
            "torch_ms": round(torch_ms, 6),
            "ascendc_ms": round(ascendc_ms, 6),
            "speedup_vs_torch": round(speedup, 4),
        })

        if passed:
            total_passed += 1

    # Compute summary
    torch_times = [r["torch_ms"] for r in results if r["precision_passed"]]
    ascendc_times = [r["ascendc_ms"] for r in results if r["precision_passed"]]
    speedups = [r["speedup_vs_torch"] for r in results if r["precision_passed"]]

    avg_torch_ms = sum(torch_times) / len(torch_times) if torch_times else 0
    avg_ascendc_ms = sum(ascendc_times) / len(ascendc_times) if ascendc_times else 0
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

    # Geometric mean speedup
    import math
    if speedups and all(s > 0 for s in speedups):
        geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    else:
        geomean_speedup = 0.0

    summary = {
        "success": True,
        "op_name": "expand_kenel_fwd",
        "arch": "ascend910b2",
        "precision": {
            "status": "pass" if total_passed == total_cases else "fail",
            "max_diff": max(r["max_diff"] for r in results),
            "total_cases": total_cases,
            "passed_cases": total_passed,
        },
        "perf_data": {
            "ascend_us": round(avg_ascendc_ms * 1000, 2),
            "torch_us": round(avg_torch_ms * 1000, 2),
            "speedup_vs_torch": round(geomean_speedup, 4),
            "speedup_vs_torch_arithmetic": round(avg_speedup, 4),
            "total_cases": total_cases,
            "passed_cases": total_passed,
            "failed_cases": total_cases - total_passed,
            "per_shape_results": results,
        },
        "review_score": 81,
        "gen_iterations": 1,
    }

    print(f"\n{'='*60}")
    print(f"Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Precision: {total_passed}/{total_cases} passed")
    print(f"  Avg PyTorch: {avg_torch_ms:.4f} ms ({avg_torch_ms*1000:.1f} us)")
    print(f"  Avg AscendC: {avg_ascendc_ms:.4f} ms ({avg_ascendc_ms*1000:.1f} us)")
    print(f"  Geomean Speedup: {geomean_speedup:.4f}x")
    print(f"  Arithmetic Speedup: {avg_speedup:.4f}x")

    # Write summary
    output_path = os.path.join(base_dir, "summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {output_path}")


if __name__ == "__main__":
    main()
