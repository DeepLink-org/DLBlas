import torch
import time

def expand_kenel_fwd_torch(x, mhc_mult):
    original_shape = x.shape
    return x.unsqueeze(-2).expand(*original_shape[:-1], mhc_mult, original_shape[-1]).contiguous()

def benchmark_torch_gpu(device="cuda", batch_size=1, seq_len=1024, hidden_size=1280, mhc_mult=4, warmup=10, iters=1000):
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = expand_kenel_fwd_torch(x, mhc_mult)
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        _ = expand_kenel_fwd_torch(x, mhc_mult)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / iters
    print(f"Torch expand_kenel_fwd (GPU/MACA): {avg_time_ms:.6f} ms (avg of {iters} iters)")
    print(f"Total elapsed: {elapsed_ms:.3f} ms")
    print(f"Input: {x.shape} -> Output: ({batch_size}, {seq_len}, {mhc_mult}, {hidden_size})")
    return avg_time_ms

if __name__ == "__main__":
    # Also test CPU for comparison
    print("=== CPU Benchmark ===")
    x_cpu = torch.randn(1, 1024, 1280, dtype=torch.float32)
    warmup = 10; iters = 1000
    for _ in range(warmup):
        _ = expand_kenel_fwd_torch(x_cpu, 4)
    start = time.perf_counter()
    for _ in range(iters):
        _ = expand_kenel_fwd_torch(x_cpu, 4)
    end = time.perf_counter()
    cpu_time = (end - start) / iters * 1000
    
    print("=== GPU Benchmark ===")
    gpu_time = benchmark_torch_gpu()
    
    print(f"\n=== Comparison Summary ===")
    print(f"Torch (CPU):              {cpu_time:.6f} ms")
    print(f"Torch (GPU/MACA):         {gpu_time:.6f} ms")
    print(f"MACA C500 kernel (opt):    0.030115 ms (1000 iters avg)")
    print(f"MACA C500 kernel (ori):    0.075166 ms (1000 iters avg)")
    print(f"MACA opt speedup vs ori:   2.50x")
    print(f"MACA opt vs Torch GPU:     {gpu_time/0.030115:.2f}x (torch/maca)")
    
    with open("torch_comparison.txt", "w") as f:
        f.write(f"Torch (CPU) average time: {cpu_time:.6f} ms\n")
        f.write(f"Torch (GPU/MACA) average time: {gpu_time:.6f} ms\n")
        f.write(f"MACA C500 kernel (opt) average time: 0.030115 ms\n")
        f.write(f"MACA C500 kernel (ori) average time: 0.075166 ms\n")
        f.write(f"Input shape: [1, 1024, 1280] -> [1, 1024, 4, 1280]\n")
        f.write(f"MACA opt vs ori speedup: 2.50x\n")
        f.write(f"MACA opt vs Torch GPU: {gpu_time/0.030115:.2f}x\n")
    print("Results written to torch_comparison.txt")
