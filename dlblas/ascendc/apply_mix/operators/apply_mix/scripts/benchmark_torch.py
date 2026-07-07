"""Benchmark PyTorch reference vs AscendC apply_mix operator on NPU."""
import sys, os, time
import torch
import torch_npu

# Load the AscendC op
SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'libapply_mix_ops.so')
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found")
    sys.exit(1)
torch.ops.load_library(SO_PATH)

# Default shape
n0, n1, mhc, h = 2, 1024, 4, 1280
WARMUP = 50
REPEAT = 200

def benchmark_ascendc(x_npu, mix_npu):
    """Benchmark AscendC operator."""
    op_fn = getattr(torch.ops.npu, "apply_mix")
    # Warmup
    for _ in range(WARMUP):
        _ = op_fn(x_npu, mix_npu)
    torch.npu.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(REPEAT):
        _ = op_fn(x_npu, mix_npu)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / REPEAT) * 1e6  # us

def benchmark_torch(x_npu, mix_npu):
    """Benchmark PyTorch reference: (x * mix).sum(-2).bfloat16()"""
    # Warmup
    for _ in range(WARMUP):
        _ = (x_npu.float() * mix_npu).sum(dim=-2).bfloat16()
    torch.npu.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(REPEAT):
        _ = (x_npu.float() * mix_npu).sum(dim=-2).bfloat16()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / REPEAT) * 1e6  # us

def main():
    print(f"=== apply_mix Benchmark: AscendC vs PyTorch (NPU) ===")
    print(f"Shape: n0={n0}, n1={n1}, mhc={mhc}, h={h}")
    print(f"Warmup: {WARMUP}, Repeat: {REPEAT}")
    
    # Generate data
    torch.manual_seed(42)
    x_bf16 = torch.sigmoid(torch.randn(n0, n1, mhc, h)).bfloat16()
    mix_fp32 = torch.nn.functional.softmax(torch.randn(n0, n1, mhc, 1), dim=-2)
    
    x_npu = x_bf16.npu()
    mix_npu = mix_fp32.npu()
    
    # Benchmark AscendC
    print("\n--- AscendC Operator ---")
    ascendc_us = benchmark_ascendc(x_npu, mix_npu)
    print(f"  Avg latency: {ascendc_us:.2f} us")
    
    # Benchmark PyTorch
    print("\n--- PyTorch Reference (NPU) ---")
    torch_us = benchmark_torch(x_npu, mix_npu)
    print(f"  Avg latency: {torch_us:.2f} us")
    
    speedup = torch_us / ascendc_us
    print(f"\n=== Results ===")
    print(f"  AscendC: {ascendc_us:.2f} us")
    print(f"  PyTorch: {torch_us:.2f} us")
    print(f"  Speedup (AscendC vs PyTorch): {speedup:.4f}x")
    
    # Also output machine-readable JSON
    import json
    result = {
        "ascendc_us": round(ascendc_us, 2),
        "torch_us": round(torch_us, 2),
        "speedup_vs_torch": round(speedup, 4)
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bench_result.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to: {out_path}")

if __name__ == "__main__":
    main()
