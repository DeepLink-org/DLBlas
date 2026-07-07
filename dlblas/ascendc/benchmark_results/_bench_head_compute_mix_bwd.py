#!/usr/bin/env python3
# Auto-generated benchmark for head_compute_mix_bwd
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

    so=f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/head_compute_mix_bwd/operators/head_compute_mix_bwd/build/libhead_compute_mix_bwd_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"head_compute_mix_bwd","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from head_compute_mix_bwd import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        a,b,c,d = data[:4]; a_n,b_n,c_n = a.npu(),b.npu(),c.npu()
        tu=bench(lambda:m.forward(a,b,c,d)); au=bench(lambda:torch.ops.npu.head_compute_mix_bwd(a_n,b_n,c_n,d))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"head_compute_mix_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"head_compute_mix_bwd","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
