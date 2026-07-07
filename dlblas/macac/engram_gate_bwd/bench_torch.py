
import sys, time, os
sys.path.insert(0,"/datapool/zmz/04kernelagent/waic/origin")
sys.path.insert(0,"/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_bwd_run")
os.chdir("/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_bwd_run")
import engram_gate_bwd
model = engram_gate_bwd.Model(*engram_gate_bwd.get_init_inputs())
inputs = engram_gate_bwd.get_inputs()
inputs = [x.cuda() if hasattr(x,'cuda') else x for x in inputs]
for _ in range(10): model.forward(*inputs)
import torch; torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(100): model.forward(*inputs)
torch.cuda.synchronize()
t1=time.perf_counter()
tm=(t1-t0)/100*1000
print(f'<torch_time>{tm:.6f} ms</torch_time>')
