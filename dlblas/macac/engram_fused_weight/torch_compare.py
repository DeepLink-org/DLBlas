import torch
import time
import numpy as np

# Same dimensions as the MACA kernel
hc_mult = 4
hidden_size = 128
size = hc_mult * hidden_size

# Check if CUDA/MACA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Create input data (matching the test initialization pattern)
wh_cpu = torch.tensor([np.sin(i * 7 * 0.1) * 0.5 for i in range(size)], dtype=torch.bfloat16)
we_cpu = torch.tensor([np.cos((i * 13 + 3) * 0.1) * 0.3 for i in range(size)], dtype=torch.bfloat16)

wh = wh_cpu.reshape(hc_mult, hidden_size).to(device)
we = we_cpu.reshape(hc_mult, hidden_size).to(device)

# Warmup
for _ in range(100):
    result = wh.float() * we.float()
if device == "cuda":
    torch.cuda.synchronize()

# Benchmark
num_iters = 1000
if device == "cuda":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        result = wh.float() * we.float()
    end.record()
    torch.cuda.synchronize()
    total_time_ms = start.elapsed_time(end)
else:
    start = time.perf_counter()
    for _ in range(num_iters):
        result = wh.float() * we.float()
    total_time_ms = (time.perf_counter() - start) * 1000.0

avg_time_ms = total_time_ms / num_iters
print(f"\n=== Torch Performance ===")
print(f"Total time ({num_iters} iterations): {total_time_ms:.3f} ms")
print(f"Average time per call: {avg_time_ms:.6f} ms")
print(f"Operation: wh.float() * we.float()")
print(f"Input shape: [{hc_mult}, {hidden_size}], dtype=bfloat16")
print(f"Output dtype: float32")
print(f"Num elements: {size}")

# Save result
with open("torch_comparison.txt", "w") as f:
    f.write(f"Torch avg time: {avg_time_ms:.6f} ms\n")
    f.write(f"Torch total time ({num_iters} iters): {total_time_ms:.3f} ms\n")
    f.write(f"Device: {device}\n")
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"Operation: wh.float() * we.float()\n")
    f.write(f"Shape: [{hc_mult}, {hidden_size}], bf16->f32\n")

print("Results saved to torch_comparison.txt")
