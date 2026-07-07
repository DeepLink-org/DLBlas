#!/usr/bin/env python3
# Auto-generated benchmark for pre_split_mixes
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/pre_split_mixes/build/libpre_split_mixes_ops.so")
    from pre_split_mixes import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        im,ms,mb,eps = data[:4]; im_n,ms_n,mb_n = im.npu(),ms.npu(),mb.npu()
        tu=bench(lambda:m.forward(im,ms,mb,eps)); au=bench(lambda:torch.ops.npu.pre_split_mixes(im_n,ms_n,mb_n,eps))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"pre_split_mixes","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"pre_split_mixes","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
