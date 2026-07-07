"""MACA vs Torch performance comparison for MTPBlock HC kernel"""
import torch
import time
import numpy as np

# Match the MACA test dimensions
B, S, HC, D = 1, 8, 4, 512
HC_D = HC * D
MIX_HC = (2 + HC) * HC
eps = 1e-6
sinkhorn_iters = 20

def hc_weight(i, j, seed=0):
    """Exact same weight function as MACA kernel"""
    v = np.sin(float(i * 127 + j * 31 + seed * 13) * 0.0174533)
    v += np.cos(float(i * 73 + j * 17 - seed * 29) * 0.0174533)
    return v * 0.01

def mtpblock_hc_torch(x):
    """PyTorch implementation matching the MACA kernel"""
    # x: [B*S, HC, D]
    rows = B * S
    x_flat = x.reshape(rows, HC_D)
    
    # RMS normalization
    rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    x_norm = x_flat * rms
    
    # Build hc_fn weight matrix [MIX_HC, HC_D]
    weight_np = np.zeros((MIX_HC, HC_D), dtype=np.float32)
    for i in range(MIX_HC):
        for j in range(HC_D):
            weight_np[i, j] = hc_weight(i, j)
    hc_fn = torch.from_numpy(weight_np)
    
    # Linear projection: mixes = x_norm @ hc_fn^T [rows, MIX_HC]
    mixes = x_norm @ hc_fn.T
    
    # Pre: sigmoid(mixes[:, :HC]) + eps
    pre = torch.sigmoid(mixes[:, :HC]) + eps
    
    # Weighted combination: y[b, d] = sum_hc pre[b, h] * x[b, h, d]
    x_shaped = x.reshape(rows, HC, D)
    y = torch.sum(pre.unsqueeze(-1) * x_shaped, dim=1)
    
    return y

# Warmup + benchmark
print("="*60)
print("MTPBlock HC Kernel — Torch Performance Reference")
print(f"Shape: B={B}, S={S}, HC={HC}, D={D}")
print(f"MIX_HC={MIX_HC}, HC_D={HC_D}")
print("="*60)

# Create test input (same pattern as MACA test)
x_data = np.array([(i * 7) % 127 for i in range(B * S * HC * D)], dtype=np.float32) * 0.01 - 0.635
x = torch.from_numpy(x_data).reshape(B, S, HC, D)

# Warmup
print("\nWarming up (10 iterations)...")
for _ in range(10):
    _ = mtpblock_hc_torch(x)
torch.cuda.synchronize() if torch.cuda.is_available() else None

# Benchmark
print("Running benchmark (500 iterations)...")
times = []
for _ in range(500):
    start = time.perf_counter()
    y = mtpblock_hc_torch(x)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # ms

avg_time = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)
std_time = np.std(times)

print(f"\n{"="*60}")
print(f"Torch CPU Results:")
print(f"  Average time: {avg_time:.6f} ms")
print(f"  Min time:     {min_time:.6f} ms")
print(f"  Max time:     {max_time:.6f} ms")
print(f"  Std dev:      {std_time:.6f} ms")
print(f"{"="*60}")

# MACA results from final rerun
maca_baseline = 1.208037  # ms
maca_best = 1.019707      # ms

print(f"\n{"="*60}")
print(f"MACA C500 Results:")
print(f"  Baseline (ori):  {maca_baseline:.6f} ms")
print(f"  Best (opt):      {maca_best:.6f} ms")
print(f"  Speedup:         {maca_baseline/maca_best:.2f}x")
print(f"{"="*60}")

print(f"\n{"="*60}")
print(f"Torch vs MACA Comparison:")
print(f"  Torch CPU avg:   {avg_time:.6f} ms")
print(f"  MACA C500 best:  {maca_best:.6f} ms")
if avg_time > 0:
    print(f"  MACA advantage:  {avg_time/maca_best:.2f}x faster")
print(f"{"="*60}")

# Also try CUDA if available
if torch.cuda.is_available():
    print("\nRunning CUDA benchmark...")
    x_cuda = x.cuda()
    # Warmup
    for _ in range(10):
        _ = mtpblock_hc_torch(x_cuda)
    torch.cuda.synchronize()
    
    # CUDA benchmark
    cuda_times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(500):
        starter.record()
        y_cuda = mtpblock_hc_torch(x_cuda)
        ender.record()
        torch.cuda.synchronize()
        cuda_times.append(starter.elapsed_time(ender))
    
    cuda_avg = np.mean(cuda_times)
    print(f"  CUDA avg time:  {cuda_avg:.6f} ms")
    print(f"  MACA advantage:  {cuda_avg/maca_best:.2f}x")
    
    # Compare output correctness
    y_cpu = mtpblock_hc_torch(x)
    max_diff = (y_cuda.cpu() - y_cpu).abs().max().item()
    print(f"  Max diff (CUDA vs CPU): {max_diff:.8f}")
