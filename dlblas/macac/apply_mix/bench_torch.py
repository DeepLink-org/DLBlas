import sys, time, os
sys.path.insert(0,"/datapool/zmz/04kernelagent/waic/origin")
import apply_mix
model = apply_mix.Model(*apply_mix.get_init_inputs())
inputs = apply_mix.get_inputs()
inputs = [x.cuda() if hasattr(x,'cuda') else x for x in inputs]
import torch
for _ in range(10): model.forward(*inputs)
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(100): model.forward(*inputs)
torch.cuda.synchronize()
t1=time.perf_counter()
print(f'<torch_time>{(t1-t0)/100*1000:.6f} ms</torch_time>')
