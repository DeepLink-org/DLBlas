#!/usr/bin/env python3
# Auto-generated benchmark for apply_mix
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/apply_mix/build/libapply_mix_ops.so")
    r=[]; cfgs=[("default",2,1024,4,1280),("small",1,128,2,640),("large_b",8,1024,4,1280),("large_s",2,4096,4,1280)]
    for nm,n0,n1,m,h in cfgs:
        torch.manual_seed(42); x=torch.sigmoid(torch.randn(n0,n1,m,h)).bfloat16().npu(); mix=torch.nn.functional.softmax(torch.randn(n0,n1,m,1),dim=-2).npu()
        tu=bench(lambda:(x.float()*mix).sum(-2).bfloat16()); au=bench(lambda:torch.ops.npu.apply_mix(x,mix))
        r.append({"label":nm,"shape":[n0,n1,m,h],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"apply_mix","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"apply_mix","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
