
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from pre_split_mixes import Model, get_inputs

torch.ops.load_library("/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/pre_split_mixes/build/libpre_split_mixes_ops.so")

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
try:
    data=get_inputs(); m=Model(4)  # mhc_mult=4
    im=data[0]; im_n=im.npu()
    tu=bench(lambda:m.forward(im)); au=bench(lambda:torch.ops.npu.pre_split_mixes(im_n,*[d.npu() if isinstance(d,torch.Tensor) else d for d in data[1:]]))
    speedup=tu/au if au>0 else 0
    r=[{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}]
except Exception as e:
    import traceback
    r=[{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"pre_split_mixes","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
