import sys,time,os
sys.path.insert(0,"/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run")
import engram_gate_fwd
model = engram_gate_fwd.Model(*engram_gate_fwd.get_init_inputs())
if hasattr(model,"cuda"): model=model.cuda()
inputs = engram_gate_fwd.get_inputs()
inputs=[x.cuda() if hasattr(x,"cuda") else x for x in inputs]
import torch
for _ in range(3): model.forward(*inputs)
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(10): model.forward(*inputs)
torch.cuda.synchronize()
t1=time.perf_counter()
print(f"<torch_time>{(t1-t0)/10*1000:.6f} ms</torch_time>")
