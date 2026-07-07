#!/usr/bin/env python3
# Auto-generated benchmark for norm_fn
import sys, os, time, json, math
import numpy as np
import torch
import torch_npu
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
WARMUP = 10; REPEAT = 100
def sync(): torch.npu.synchronize()
def bench(fn, w=WARMUP, r=REPEAT):
    for _ in range(w): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6
def geomean(vals):
    v = [x for x in vals if x > 0]
    return math.exp(sum(math.log(x) for x in v)/len(v)) if v else 0.0
try:

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/norm_fn/build/libnorm_fn_ops.so")
    from norm_fn import Model, generate_norm_fn_test_data
    r=[]
    for nm,n1,mh,hs in [("default",13,4,1280),("small",1,2,640),("medium",26,4,1280)]:
        try:
            res,fn,nw,og,eps=generate_norm_fn_test_data(n1,mh,hs,False); m=Model()
            res_n,fn_n=res.npu(),fn.npu(); nw_n=nw.npu() if nw is not None else None
            tu=bench(lambda:m.forward(res,fn,nw,eps)); au=bench(lambda:torch.ops.npu.norm_fn(res_n,fn_n,nw_n,eps))
            r.append({"label":nm,"shape":[n1,mh,hs],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[n1,mh,hs],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"norm_fn","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"norm_fn","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
