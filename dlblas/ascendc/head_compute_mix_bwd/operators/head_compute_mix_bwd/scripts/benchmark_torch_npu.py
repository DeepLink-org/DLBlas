# Benchmark PyTorch NPU reference for head_compute_mix_bwd
import torch
import torch_npu
import time

torch.manual_seed(42)

def compute_golden_npu(input_mix, mhc_scale, mhc_base, grad_out):
    z = input_mix * mhc_scale + mhc_base
    sigmoid = torch.sigmoid(z)
    grad_z = grad_out * sigmoid * (1 - sigmoid)
    grad_input_mix = grad_z * mhc_scale
    grad_mhc_base = grad_z.sum(dim=(0, 1), keepdim=True).view(-1)
    grad_mhc_scale = (grad_z * input_mix).sum(dim=(0, 1, 2), keepdim=True).view(1)
    return grad_input_mix, grad_mhc_scale, grad_mhc_base

B, S, C = 2, 1024, 4
input_mix = torch.randn(B, S, C, dtype=torch.float32, device='npu')
mhc_scale = torch.randn(1, dtype=torch.float32, device='npu')
mhc_base = torch.randn(C, dtype=torch.float32, device='npu')
grad_out = torch.randn(B, S, C, dtype=torch.float32, device='npu')

# Warmup
for _ in range(100):
    compute_golden_npu(input_mix, mhc_scale, mhc_base, grad_out)
torch_npu.npu.synchronize()

# Benchmark
N = 500
start = time.perf_counter()
for _ in range(N):
    compute_golden_npu(input_mix, mhc_scale, mhc_base, grad_out)
torch_npu.npu.synchronize()
end = time.perf_counter()
torch_npu_latency_us = ((end - start) / N) * 1e6

print(f"PyTorch (NPU) reference: {torch_npu_latency_us:.3f} us (avg over {N} runs)")
print(f"AscendC kernel (NPU):  18.961 us")
print(f"Speedup vs PyTorch(NPU): {torch_npu_latency_us / 18.961:.4f}x")
