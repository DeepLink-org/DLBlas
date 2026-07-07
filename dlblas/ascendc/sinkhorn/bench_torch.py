import time
import torch
import torch.nn as nn
import numpy as np

# Same Model as origin/sinkhorn.py
class Model(nn.Module):
    def __init__(self, repeat=10, eps=1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps
    def forward(self, x):
        x = x.softmax(-1) + self.eps
        x = x / (x.sum(-2, keepdim=True) + self.eps)
        for _ in range(self.repeat - 1):
            x = x / (x.sum(-1, keepdim=True) + self.eps)
            x = x / (x.sum(-2, keepdim=True) + self.eps)
        return x

# Warm-up and benchmark on NPU
model = Model(repeat=10, eps=1e-6).npu()
model.eval()

x = torch.randn(1, 1024, 4, 4).npu()

# Warm-up
for _ in range(10):
    with torch.no_grad():
        _ = model(x)
torch.npu.synchronize()

# Benchmark
n_warmup = 5
n_iter = 100
times = []

for i in range(n_warmup + n_iter):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    if i >= n_warmup:
        times.append((t1 - t0) * 1e6)  # convert to us

times = np.array(times)
mean_us = np.mean(times)
median_us = np.median(times)
min_us = np.min(times)
std_us = np.std(times)

print(f"PyTorch Sinkhorn Benchmark (NPU, {n_iter} iters)")
print(f"  Shape: [1, 1024, 4, 4] float32")
print(f"  Mean:   {mean_us:.2f} us")
print(f"  Median: {median_us:.2f} us")
print(f"  Min:    {min_us:.2f} us")
print(f"  Std:    {std_us:.2f} us")

# Output JSON for easy parsing
import json
result = {
    "mean_us": round(mean_us, 2),
    "median_us": round(median_us, 2),
    "min_us": round(min_us, 2),
    "std_us": round(std_us, 2),
    "n_iter": n_iter
}
print(f"\nJSON: {json.dumps(result)}")
