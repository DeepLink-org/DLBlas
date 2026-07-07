"""PyTorch NPU baseline benchmark for engram_gate_w_reduce operator."""
import torch
import time
import os
import json

def benchmark_pytorch_npu(hidden_size=4096, warmup=10, iters=100):
    """Run the PyTorch reference implementation on NPU and measure latency."""
    hc_mult = 4
    num_persistent_blocks = 108

    # Use NPU device
    device = 'npu'
    grad_w_partial = torch.randn(num_persistent_blocks, hc_mult, hidden_size,
                                 dtype=torch.float32, device=device)
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    grad_weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)

    def forward():
        grad_w_sum = grad_w_partial.sum(0)
        grad_weight_hidden_out = grad_weight_hidden + grad_w_sum * weight_embed.float()
        grad_weight_embed_out = grad_weight_embed + grad_w_sum * weight_hidden.float()
        return grad_weight_hidden_out, grad_weight_embed_out

    # Warmup
    for _ in range(warmup):
        forward()
    torch.npu.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        forward()
    torch.npu.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    avg_us = avg_ms * 1000
    return avg_us, avg_ms


def benchmark_pytorch_cpu(hidden_size=4096, warmup=10, iters=50):
    """Run the PyTorch reference implementation on CPU and measure latency."""
    hc_mult = 4
    num_persistent_blocks = 108
    device = 'cpu'

    grad_w_partial = torch.randn(num_persistent_blocks, hc_mult, hidden_size,
                                 dtype=torch.float32, device=device)
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    grad_weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)

    def forward():
        grad_w_sum = grad_w_partial.sum(0)
        grad_weight_hidden_out = grad_weight_hidden + grad_w_sum * weight_embed.float()
        grad_weight_embed_out = grad_weight_embed + grad_w_sum * weight_hidden.float()
        return grad_weight_hidden_out, grad_weight_embed_out

    for _ in range(warmup):
        forward()

    start = time.perf_counter()
    for _ in range(iters):
        forward()
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    avg_us = avg_ms * 1000
    return avg_us, avg_ms


def main():
    hs = int(os.environ.get("HIDDEN_SIZE", 4096))

    print("=== PyTorch Performance Benchmark ===")
    print(f"  hidden_size={hs}")

    # NPU benchmark
    print("\n--- NPU ---")
    try:
        npu_us, npu_ms = benchmark_pytorch_npu(hs, warmup=10, iters=50)
        print(f"  PyTorch NPU: {npu_us:.2f} us ({npu_ms:.4f} ms)")
    except Exception as e:
        print(f"  NPU failed: {e}")
        npu_us, npu_ms = None, None

    # CPU benchmark
    print("\n--- CPU ---")
    try:
        cpu_us, cpu_ms = benchmark_pytorch_cpu(hs, warmup=5, iters=20)
        print(f"  PyTorch CPU: {cpu_us:.2f} us ({cpu_ms:.4f} ms)")
    except Exception as e:
        print(f"  CPU failed: {e}")
        cpu_us, cpu_ms = None, None

    result = {
        "torch_npu_latency_us": round(npu_us, 2) if npu_us else None,
        "torch_npu_latency_ms": round(npu_ms, 4) if npu_ms else None,
        "torch_cpu_latency_us": round(cpu_us, 2) if cpu_us else None,
        "torch_cpu_latency_ms": round(cpu_ms, 4) if cpu_ms else None,
        "hidden_size": hs
    }
    print(f"\nResult: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
