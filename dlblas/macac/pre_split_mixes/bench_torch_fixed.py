import sys, time, os
sys.path.insert(0, "/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run")
import pre_split_mixes
import torch

# Create model and move to cuda
model = pre_split_mixes.Model(*pre_split_mixes.get_init_inputs())
model = model.cuda()

# Get inputs and move to cuda
inputs = pre_split_mixes.get_inputs()
inputs = [x.cuda() if hasattr(x, "cuda") else x for x in inputs]

# Warmup
for _ in range(20):
    model.forward(*inputs)
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
for _ in range(500):
    model.forward(*inputs)
torch.cuda.synchronize()
t1 = time.perf_counter()

torch_time_ms = (t1 - t0) / 500 * 1000
print(f"<torch_time>{torch_time_ms:.6f} ms</torch_time>")
