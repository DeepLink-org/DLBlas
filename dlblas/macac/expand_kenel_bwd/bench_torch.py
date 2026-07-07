import sys,time,os
sys.path.insert(0,"/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/expand_kenel_bwd_run")
import expand_kenel_bwd
model = expand_kenel_bwd.Model(*expand_kenel_bwd.get_init_inputs())
if hasattr(model,"cuda"): model=model.cuda()
inputs = expand_kenel_bwd.get_inputs()
inputs=[x.cuda() if hasattr(x,"cuda") else x for x in inputs]
import torch
for _ in range(10): model.forward(*inputs)
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(100): model.forward(*inputs)
torch.cuda.synchronize()
t1=time.perf_counter()
print(f"<torch_time>{(t1-t0)/100*1000:.6f} ms</torch_time>")
