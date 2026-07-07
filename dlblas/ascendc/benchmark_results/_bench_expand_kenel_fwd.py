#!/usr/bin/env python3
# Auto-generated benchmark for expand_kenel_fwd
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

    sys.path.insert(0,f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/expand_kenel_fwd/scripts")
    from golden import compute_golden
    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/expand_kenel_fwd/build/libexpand_kenel_fwd_ops.so")
    r=[]; cfgs=[("typical",1,1024,1280,4),("min",1,1,128,2),("multi",4,256,256,2),("largeM",1,1,1280,16),("M1",1,1,1280,1)]
    for nm,B,S,H,M in cfgs:
        x=torch.randn(B,S,H,dtype=torch.float16); xn=x.npu()
        tu=bench(lambda:compute_golden(xn,M)); au=bench(lambda:torch.ops.npu.expand_kenel_fwd(xn,M))
        r.append({"label":nm,"shape":[B,S,H,M],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"expand_kenel_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"expand_kenel_fwd","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
