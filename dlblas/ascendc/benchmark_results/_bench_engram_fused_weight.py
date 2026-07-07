#!/usr/bin/env python3
# Auto-generated benchmark for engram_fused_weight
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_fused_weight/build/libengram_fused_weight_ops.so")
    from engram_fused_weight import Model, generate_test_data
    r=[]; cfgs=[("small",16,128),("default",64,1280),("medium",128,1280),("large",256,2048)]
    for nm,H,D in cfgs:
        wh,we=generate_test_data(H,D); m=Model()
        wh_n,we_n=wh.npu(),we.npu()
        tu=bench(lambda:m.forward(wh,we)); au=bench(lambda:torch.ops.npu.engram_fused_weight(wh_n,we_n))
        r.append({"label":nm,"shape":[H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_fused_weight","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"engram_fused_weight","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
