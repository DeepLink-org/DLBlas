
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from big_fuse import Model, get_inputs

so="/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/big_fuse/operators/big_fuse/build/libbig_fuse_ops.so"
if not os.path.exists(so):
    print(json.dumps({"op":"big_fuse","error":"no .so","status":"SKIP"}))
    sys.exit(0)
torch.ops.load_library(so)

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

r=[]
try:
    data=get_inputs(); m=Model(4, 1280)  # mhc_mult=4, hidden_size=1280
    residual=data[0]; residual_n=residual.npu()
    # Model.forward takes (self, residual) - returns tuple
    # torch.ops.npu.big_fuse expects (residual, fn, mhc_scale, mhc_base) -> Tensor[]
    # get_inputs returns: residual, fn, mhc_scale, mhc_base
    tu=bench(lambda:m.forward(residual))
    fn_n,ms_n,mb_n = data[1].npu(),data[2].npu(),data[3].npu()
    au=bench(lambda:torch.ops.npu.big_fuse(residual_n,fn_n,ms_n,mb_n))
    speedup=tu/au if au>0 else 0
    r=[{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}]
except Exception as e:
    import traceback
    r=[{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"big_fuse","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
