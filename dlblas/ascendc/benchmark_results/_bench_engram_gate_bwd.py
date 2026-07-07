#!/usr/bin/env python3
# Auto-generated benchmark for engram_gate_bwd
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_gate_bwd/build/libengram_gate_bwd_ops.so")
    from engram_gate_bwd import Model, generate_test_data
    r=[]; cfgs=[("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128),("T64H4D128",64,4,128)]
    for nm,T,H,D in cfgs:
        try:
            go,x,k,v,wh,we=generate_test_data(T,H,D); m=Model()
            go_n,x_n,k_n,v_n,wh_n,we_n=go.npu(),x.npu(),k.npu(),v.npu(),wh.npu(),we.npu()
            tu=bench(lambda:m.forward(go,x,k,v,wh,we)); au=bench(lambda:torch.ops.npu.engram_gate_bwd(go_n,x_n,k_n,v_n,wh_n,we_n))
            r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[T,H,D],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_gate_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"engram_gate_bwd","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
