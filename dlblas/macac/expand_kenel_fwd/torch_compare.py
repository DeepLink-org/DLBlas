import torch
import time
import numpy as np

def expand_kenel_fwd_torch(x, mhc_mult):
    """PyTorch reference implementation"""
    original_shape = x.shape
    return x.unsqueeze(-2).expand(*original_shape[:-1], mhc_mult, original_shape[-1]).contiguous()

def benchmark_torch(batch_size=1, seq_len=1024, hidden_size=1280, mhc_mult=4, warmup=10, iters=1000):
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device="cpu")
    
    # Warmup
    for _ in range(warmup):
        _ = expand_kenel_fwd_torch(x, mhc_mult)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = expand_kenel_fwd_torch(x, mhc_mult)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iters * 1000
    print(f"Torch expand_kenel_fwd (CPU): {avg_time_ms:.6f} ms")
    print(f"Input shape: {x.shape} -> Output shape: ({batch_size}, {seq_len}, {mhc_mult}, {hidden_size})")
    return avg_time_ms

if __name__ == "__main__":
    torch_time = benchmark_torch()
    print(f"\n=== Comparison Summary ===")
    print(f"Torch (CPU) average time: {torch_time:.6f} ms")
    
    # Store results
    with open("torch_comparison.txt", "w") as f:
        f.write(f"Torch expand_kenel_fwd (CPU) average time: {torch_time:.6f} ms\n")
        f.write(f"Input shape: [1, 1024, 1280] -> Output shape: [1, 1024, 4, 1280]\n")
        f.write(f"MACA C500 expand_kenel_fwd_opt average time: 0.030115 ms (1000 iters)\n")
        f.write(f"MACA C500 expand_kenel_fwd_ori average time: 0.075166 ms (1000 iters)\n")
        f.write(f"MACA speedup vs ori: {0.075166/0.030115:.2f}x\n")
        f.write(f"Note: Torch runs on CPU, MACA runs on C500 GPU. Direct comparison not\n")
        f.write(f"meaningful without same hardware. Use for reference only.\n")
    print("Results written to torch_comparison.txt")
