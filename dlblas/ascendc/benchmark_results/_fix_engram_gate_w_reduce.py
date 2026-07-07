
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from engram_gate_w_reduce import Model, generate_test_data

so="/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_gate_w_reduce/operators/engram_gate_w_reduce/build/libengram_gate_w_reduce_ops.so"
if not os.path.exists(so):
    print(json.dumps({"op":"engram_gate_w_reduce","error":"no .so","status":"SKIP"}))
    sys.exit(0)
torch.ops.load_library(so)

WARMUP=10; REPEAT=100
def sync(): torch.npu.synchronize()
def bench(fn,w=WARMUP,r=REPEAT):
    for _ in range(w): fn()
    sync(); t0=time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6

def geomean(v):
    vv=[x for x in v if x>0]
    return math.exp(sum(math.log(x) for x in vv)/len(vv)) if vv else 0

# Model.forward(self, grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
r=[]
for nm,D in [("default",128),("small",64)]:
    try:
        data=generate_test_data(D); m=Model()
        data_npu = [d.npu() if isinstance(d,torch.Tensor) else d for d in data]
        tu=bench(lambda:m.forward(*data)); au=bench(lambda:torch.ops.npu.engram_gate_w_reduce(*data_npu))
        r.append({"label":nm,"shape":[D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    except Exception as e:
        import traceback
        r.append({"label":nm,"shape":[D],"error":str(e),"status":"FAIL"})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"engram_gate_w_reduce","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
