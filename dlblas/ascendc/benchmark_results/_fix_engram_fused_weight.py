
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from engram_fused_weight import Model, generate_test_data

torch.ops.load_library("/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_fused_weight/build/libengram_fused_weight_ops.so")

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

r=[]
for nm,H,D in [("small",16,128),("default",64,1280),("medium",128,1280)]:
    try:
        wh,we=generate_test_data(H,D); m=Model(H,D)
        wh_n,we_n=wh.npu(),we.npu()
        tu=bench(lambda:m.forward(wh,we)); au=bench(lambda:torch.ops.npu.engram_fused_weight(wh_n,we_n))
        r.append({"label":nm,"shape":[H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    except Exception as e:
        r.append({"label":nm,"shape":[H,D],"error":str(e),"status":"FAIL"})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"engram_fused_weight","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
