import torch
import time
import subprocess
import re
import numpy as np

# ===== Torch Reference Implementation =====
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref):
        grad_w_sum = grad_w_partial.sum(0)
        grad_wh_out = grad_wh_ref + grad_w_sum * weight_embed.float()
        grad_we_out = grad_we_ref + grad_w_sum * weight_hidden.float()
        return grad_wh_out, grad_we_out

def generate_test_data(hidden_size, device='cuda'):
    hc_mult = 4
    num_persistent_blocks = 108
    grad_w_partial = torch.randn(num_persistent_blocks, hc_mult, hidden_size, dtype=torch.float32, device=device)
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    grad_wh_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_we_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    return grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref

print("="*60)
print("engram_gate_w_reduce: Torch vs MACAC Performance Comparison")
print("="*60)

device = 'cuda'
hidden_size = 4096
warmup = 10
iters = 100

grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref = generate_test_data(hidden_size, device)

model = Model().to(device)

# Warmup
for _ in range(warmup):
    _ = model(grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref)
torch.cuda.synchronize()

# Torch benchmark
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(iters):
    _ = model(grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref)
torch.cuda.synchronize()
end = time.perf_counter()
torch_time_ms = (end - start) / iters * 1000

print(f"\n[Torch]   Average time: {torch_time_ms:.6f} ms")

# ===== MACAC Kernel Benchmark =====
import subprocess, os
os.chdir('/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run')

# Run MACAC kernel
result = subprocess.run(
    ['bash', '-c', 'export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0'],
    capture_output=True, text=True, cwd='/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run'
)

# Parse output
output = result.stdout + result.stderr
time_before_match = re.search(r'<time_before_opt>([\d.]+) ms</time_before_opt>', output)
time_after_match = re.search(r'<time_after_opt>([\d.]+) ms</time_after_opt>', output)
ratio_match = re.search(r'<runtime_ratio>\s*([\d.]+)</runtime_ratio>', output)
precision_match = re.search(r'<precision>(True|False)</precision>', output)

macac_ori_ms = float(time_before_match.group(1)) if time_before_match else 0.0
macac_opt_ms = float(time_after_match.group(1)) if time_after_match else 0.0
ratio_val = float(ratio_match.group(1)) if ratio_match else 0.0
precision_val = precision_match.group(1) if precision_match else 'Unknown'

print(f"[MACAC]   Original kernel: {macac_ori_ms:.6f} ms")
print(f"[MACAC]   Optimized kernel: {macac_opt_ms:.6f} ms")
print(f"[MACAC]   runtime_ratio: {ratio_val:.4f}")
print(f"[MACAC]   precision: {precision_val}")
print()

# Comparison
print("="*60)
print("Comparison Summary")
print("="*60)
print(f"{'Metric':<30} {'Torch':>15} {'MACAC-opt':>15} {'Ratio':>10}")
print("-"*70)
print(f"{'Kernel time (ms)':<30} {torch_time_ms:>15.6f} {macac_opt_ms:>15.6f} {macac_opt_ms/torch_time_ms:>10.4f}")
print(f"{'vs MACAC-ori':<30} {'-':>15} {macac_ori_ms:>15.6f} {ratio_val:>10.4f}")
print()

if macac_opt_ms > 0:
    print(f"Torch is {macac_opt_ms/torch_time_ms:.2f}x vs MACAC-opt time")
    print(f"MACAC-opt is {torch_time_ms/macac_opt_ms:.2f}x vs Torch time")

# Save results
with open('/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run/benchmark_results.txt', 'w') as f:
    f.write(f"engram_gate_w_reduce: Torch vs MACAC Performance Comparison\n")
    f.write(f"Torch version: {torch.__version__}\n")
    f.write(f"Device: MetaX C500\n")
    f.write(f"Shape: grad_w_partial[108,4,4096], weights[4,4096]\n")
    f.write(f"Warmup: {warmup}, Iterations: {iters}\n")
    f.write(f"\n")
    f.write(f"Torch time: {torch_time_ms:.6f} ms\n")
    f.write(f"MACAC original (baseline) time: {macac_ori_ms:.6f} ms\n")
    f.write(f"MACAC optimized time: {macac_opt_ms:.6f} ms\n")
    f.write(f"MACAC runtime_ratio (opt/ori): {ratio_val:.4f}\n")
    f.write(f"MACAC precision: {precision_val}\n")
    f.write(f"Torch vs MACAC-opt ratio: {torch_time_ms/macac_opt_ms:.4f}\n")

print("Results saved to benchmark_results.txt")
