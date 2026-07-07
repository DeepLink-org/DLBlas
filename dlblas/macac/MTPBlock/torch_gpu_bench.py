import torch
import time
import numpy as np
import sys
sys.path.insert(0, '.')

B, S, HC, D = 1, 8, 4, 512
HC_D = HC * D
MIX_HC = (2 + HC) * HC
eps = 1e-6

def hc_weight(i, j, seed=0):
    v = np.sin(float(i * 127 + j * 31 + seed * 13) * 0.0174533)
    v += np.cos(float(i * 73 + j * 17 - seed * 29) * 0.0174533)
    return v * 0.01

rows = B * S
x = torch.randn(rows * HC_D, dtype=torch.float32, device='cuda').reshape(rows, HC_D)

weight_np = np.zeros((MIX_HC, HC_D), dtype=np.float32)
for i in range(MIX_HC):
    for j in range(HC_D):
        weight_np[i, j] = hc_weight(i, j)

hc_fn = torch.from_numpy(weight_np).cuda()

def forward(x):
    rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    x_norm = x * rms
    return x_norm @ hc_fn.T

# Warmup
for _ in range(10):
    forward(x)
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
for _ in range(500):
    forward(x)
torch.cuda.synchronize()
t1 = time.perf_counter()
tm = (t1 - t0) / 500 * 1000
print(f"<torch_time>{tm:.6f} ms</torch_time>")
