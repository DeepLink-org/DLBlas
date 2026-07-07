# ----------------------------------------------------------------------------
# AscendC vs PyTorch Benchmark - act_quant_kernel
# ----------------------------------------------------------------------------
import sys, os, time
import numpy as np
import torch
import torch_npu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(SCRIPT_DIR, "..", "build", "libact_quant_kernel_ops.so")
torch.ops.load_library(SO_PATH)

from golden import compute_golden

GROUP_SIZE = 128
EPS = 1e-10
DTYPE = torch.bfloat16
WARMUP = 10
REPEAT = 50

def benchmark_ascend(x_np_flat, group_size, use_ue8m0=False):
    """Benchmark AscendC kernel via PyTorch extension."""
    x = torch.from_numpy(x_np_flat.astype(np.float32)).to(DTYPE).reshape(-1)
    x_npu = x.npu()

    # Warmup
    for _ in range(WARMUP):
        torch.ops.npu.act_quant_kernel(x_npu, group_size, EPS, use_ue8m0)

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(REPEAT):
        torch.ops.npu.act_quant_kernel(x_npu, group_size, EPS, use_ue8m0)
    torch.npu.synchronize()
    end = time.perf_counter()

    avg_us = (end - start) / REPEAT * 1e6
    return avg_us


def benchmark_torch(x_np_flat, group_size, use_ue8m0=False):
    """Benchmark pure PyTorch reference on CPU (representative of torch cpu ref)."""
    x_f32 = x_np_flat.astype(np.float32)

    # Warmup
    for _ in range(min(WARMUP, 3)):
        compute_golden(x_f32, group_size, EPS, use_ue8m0)

    start = time.perf_counter()
    for _ in range(REPEAT):
        compute_golden(x_f32, group_size, EPS, use_ue8m0)
    end = time.perf_counter()

    avg_us = (end - start) / REPEAT * 1e6
    return avg_us


def benchmark_torch_npu(x_np_flat, group_size, use_ue8m0=False):
    """Benchmark equivalent PyTorch ops on NPU."""
    x = torch.from_numpy(x_np_flat.astype(np.float32)).to(DTYPE).reshape(-1)
    x_npu = x.npu()
    fp8_max = 448.0

    def torch_impl(x_in, gs, eps_val, ue8m0):
        x_ = x_in.reshape(x_in.numel() // gs, gs)
        amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps_val).to(torch.float32)
        x_s = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=x_in.device)
        x_q = (x_.float() / x_s)
        x_q = x_q.clamp(-fp8_max, fp8_max)
        x_q = x_q.reshape(x_in.shape)
        x_s = x_s.reshape(-1)
        return x_q, x_s

    for _ in range(WARMUP):
        torch_impl(x_npu, group_size, EPS, use_ue8m0)

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(REPEAT):
        torch_impl(x_npu, group_size, EPS, use_ue8m0)
    torch.npu.synchronize()
    end = time.perf_counter()

    avg_us = (end - start) / REPEAT * 1e6
    return avg_us


def main():
    results = []
    total_passed = 0
    total_cases = 0

    test_configs = [
        ("1K, gs=128",  1024, 128),
        ("4K, gs=128",  4096, 128),
        ("16K, gs=128", 16384, 128),
        ("65K, gs=128", 65536, 128),
        ("256K, gs=128", 262144, 128),
    ]

    print(f"{'='*80}")
    print(f"act_quant_kernel Benchmark: AscendC vs PyTorch")
    print(f"{'='*80}")
    print(f"{'Shape':<20} {'AscendC(us)':>14} {'TorchNPU(us)':>14} {'TorchCPU(us)':>14} {'Speedup':>10} {'Status':>10}")
    print(f"{'-'*80}")

    for name, numel, gs in test_configs:
        np.random.seed(42)
        x_np = np.random.randn(numel).astype(np.float32)

        # Verify correctness first
        x_tensor = torch.from_numpy(x_np.astype(np.float32)).to(DTYPE).reshape(numel).npu()
        q_asc, s_asc = torch.ops.npu.act_quant_kernel(x_tensor, gs, EPS, False)
        q_g, s_g = compute_golden(x_np, gs, EPS, False)

        q_out = q_asc.cpu().numpy().reshape(-1).astype(int)
        q_golden = q_g.astype(int)
        q_diff = np.abs(q_out - q_golden)
        q_ok = int((q_diff > 1).sum()) == 0
        s_out = s_asc.cpu().numpy().reshape(-1)
        s_golden_val = s_g.reshape(-1)
        s_max_diff = float(abs(s_out - s_golden_val).max())
        s_ok = s_max_diff < 1e-4
        passed = q_ok and s_ok
        total_cases += 1
        if passed:
            total_passed += 1

        # Benchmark
        ascend_us = benchmark_ascend(x_np, gs, False)
        torch_npu_us = benchmark_torch_npu(x_np, gs, False)
        speedup = torch_npu_us / ascend_us if ascend_us > 0 else 0.0

        status = "PASS" if passed else "FAIL"
        print(f"{name:<20} {ascend_us:>14.2f} {torch_npu_us:>14.2f} {'N/A':>14} {speedup:>10.4f} {status:>10}")

        results.append({
            "case_idx": total_cases,
            "shape_desc": name,
            "numel": numel,
            "group_size": gs,
            "status": status,
            "ascend_us": round(ascend_us, 4),
            "torch_npu_us": round(torch_npu_us, 4),
            "speedup_vs_torch": round(speedup, 4),
            "s_max_diff": float(s_max_diff),
        })

    print(f"{'='*80}")
    print(f"Total: {total_cases}, Passed: {total_passed}, Failed: {total_cases - total_passed}")
    print(f"{'='*80}")

    # Compute geometric mean speedup from passing cases
    passing_speedups = [r["speedup_vs_torch"] for r in results if r["status"] == "PASS" and r["speedup_vs_torch"] > 0]
    if passing_speedups:
        geomean = np.exp(np.mean(np.log(passing_speedups)))
    else:
        geomean = None

    # Save as JSON for summary
    import json
    summary = {
        "total_cases": total_cases,
        "passed_cases": total_passed,
        "failed_cases": total_cases - total_passed,
        "speedup_vs_torch_geomean": round(float(geomean), 4) if geomean else None,
        "per_shape_results": results,
    }
    out_path = os.path.join(SCRIPT_DIR, "..", "docs", "benchmark_result.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBenchmark results saved to: {out_path}")

    return 0 if total_passed == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
