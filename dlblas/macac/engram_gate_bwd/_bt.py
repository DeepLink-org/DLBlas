
import sys, time, os
sys.path.insert(0, "/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_bwd_run")
os.chdir("/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_bwd_run")
import engram_gate_bwd
model = engram_gate_bwd.Model(*engram_gate_bwd.get_init_inputs())
if hasattr(model, "cuda"):
    model = model.cuda()
inputs = engram_gate_bwd.get_inputs()
inputs = [x.cuda() if hasattr(x, "cuda") else x for x in inputs]
import torch
for _ in range(5):
    try:
        model.forward(*inputs)
    except Exception as e:
        print(f"ERROR: {e}")
        break
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    try:
        model.forward(*inputs)
    except:
        break
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"<torch_time>{(t1-t0)/50*1000:.6f} ms</torch_time>")
