
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from engram_gate_bwd import Model, generate_test_data

torch.ops.load_library("/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_gate_bwd/build/libengram_gate_bwd_ops.so")

WARMUP=5; REPEAT=50
def sync(): torch.npu.synchronize()
def bench(fn,w=WARMUP,r=REPEAT):
    for _ in range(w): fn()
    sync(); t0=time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6

def geomean(v):
    vv=[x for x in v if x>0]
    return math.exp(sum(math.log(x) for x in vv)/len(vv)) if vv else 0

# Check signatures first
import inspect
sig_fwd = inspect.signature(Model.forward)
n_params = len(sig_fwd.parameters) - 1  # minus self
print(f"Model.forward takes {n_params} args: {list(sig_fwd.parameters.keys())}", file=sys.stderr)

r=[]
for nm,T,H,D in [("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128)]:
    try:
        data=generate_test_data(T,H,D)
        print(f"generate_test_data returned {len(data)} items", file=sys.stderr)
        m=Model()
        # Move data to NPU
        data_npu = [d.npu() if isinstance(d,torch.Tensor) else d for d in data]
        tu=bench(lambda:m.forward(*data))
        au=bench(lambda:torch.ops.npu.engram_gate_bwd(*data_npu))
        r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    except Exception as e:
        import traceback
        r.append({"label":nm,"shape":[T,H,D],"error":str(e),"status":"FAIL"})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"engram_gate_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
