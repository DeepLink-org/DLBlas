import torch
import time

def expand_kenel_fwd_torch(x, mhc_mult):
    """PyTorch reference: unsqueeze(-2) + expand + contiguous"""
    original_shape = x.shape
    return x.unsqueeze(-2).expand(*original_shape[:-1], mhc_mult, original_shape[-1]).contiguous()

def benchmark_torch_gpu(batch_size=1, seq_len=1024, hidden_size=1280, mhc_mult=4, warmup=10, iters=1000):
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device="cuda")

    for _ in range(warmup):
        y = expand_kenel_fwd_torch(x, mhc_mult)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        y = expand_kenel_fwd_torch(x, mhc_mult)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / iters * 1000
    return avg_time_ms

def benchmark_torch_cpu(batch_size=1, seq_len=1024, hidden_size=1280, mhc_mult=4, warmup=10, iters=1000):
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device="cpu")

    for _ in range(warmup):
        _ = expand_kenel_fwd_torch(x, mhc_mult)

    start = time.perf_counter()
    for _ in range(iters):
        _ = expand_kenel_fwd_torch(x, mhc_mult)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iters * 1000
    return avg_time_ms

if __name__ == "__main__":
    print("=" * 60)
    print("expand_kenel_fwd Performance Comparison")
    print("=" * 60)

    # Torch GPU (MACA backend)
    print("\n[1/3] Benchmarking Torch on MACA GPU...")
    torch_gpu_time = benchmark_torch_gpu()
    print("  Torch GPU (MACA backend): %.6f ms" % torch_gpu_time)

    # Torch CPU
    print("\n[2/3] Benchmarking Torch on CPU...")
    torch_cpu_time = benchmark_torch_cpu()
    print("  Torch CPU: %.6f ms" % torch_cpu_time)

    # MACA kernel times (from bench results)
    maca_opt_time = 0.030694  # Final rerun result
    maca_ori_time = 0.099453  # Final rerun baseline

    print("\n[3/3] MACA C500 Hand-written Kernel (from bench)")
    print("  MACA opt kernel: %.6f ms" % maca_opt_time)
    print("  MACA ori kernel: %.6f ms" % maca_ori_time)

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print("%-30s %-15s %-15s" % ("Implementation", "Time (ms)", "vs MACA opt"))
    print("-" * 60)
    print("%-30s %-15.6f %-15.2fx slower" % ("Torch GPU (MACA)", torch_gpu_time, torch_gpu_time/maca_opt_time))
    print("%-30s %-15.6f %-15.2fx slower" % ("Torch CPU", torch_cpu_time, torch_cpu_time/maca_opt_time))
    print("%-30s %-15.6f %-15.2fx slower" % ("MACA ori kernel", maca_ori_time, maca_ori_time/maca_opt_time))
    print("%-30s %-15.6f %-15.2fx (baseline)" % ("MACA opt kernel", maca_opt_time, 1.0))

    speedup_vs_torch = torch_gpu_time / maca_opt_time
    speedup_vs_ori = maca_ori_time / maca_opt_time
    print("\nMACA opt vs Torch GPU: %.2fx %s" % (speedup_vs_torch, "faster" if speedup_vs_torch > 1 else "slower"))
    print("MACA opt vs MACA ori: %.2fx faster" % speedup_vs_ori)

    # Write results to file
    with open("/mnt/opt_test/expand_kenel_fwd_run/torch_comparison.txt", "w") as f:
        f.write("expand_kenel_fwd Performance Comparison\n")
        f.write("=" * 60 + "\n")
        f.write("Date: 2026-06-29\n")
        f.write("Container: metax_gemm_opt\n")
        f.write("Torch version: %s\n" % torch.__version__)
        f.write("Input shape: [1, 1024, 1280] -> Output shape: [1, 1024, 4, 1280]\n\n")
        f.write("Torch GPU (MACA backend): %.6f ms\n" % torch_gpu_time)
        f.write("Torch CPU: %.6f ms\n" % torch_cpu_time)
        f.write("MACA opt kernel: %.6f ms\n" % maca_opt_time)
        f.write("MACA ori kernel: %.6f ms\n" % maca_ori_time)
        f.write("\nMACA opt vs Torch GPU: %.2fx\n" % speedup_vs_torch)
        f.write("MACA opt vs MACA ori: %.2fx faster\n" % speedup_vs_ori)

    print("\nResults saved to torch_comparison.txt")
