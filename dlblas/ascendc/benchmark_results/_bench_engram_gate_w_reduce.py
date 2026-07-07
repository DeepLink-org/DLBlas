#!/usr/bin/env python3
# Auto-generated benchmark for engram_gate_w_reduce
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

    so=f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_gate_w_reduce/operators/engram_gate_w_reduce/build/libengram_gate_w_reduce_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"engram_gate_w_reduce","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from engram_gate_w_reduce import Model, generate_test_data
    r=[]
    for nm,T,H,D in [("default",14,4,128),("small",1,2,64)]:
        try:
            data=generate_test_data(D); m=Model()
            if len(data)==6:
                gw,wh,we,x,k,v=data; gw_n,wh_n,we_n,x_n,k_n,v_n=gw.npu(),wh.npu(),we.npu(),x.npu(),k.npu(),v.npu()
                tu=bench(lambda:m.forward(gw,wh,we,x,k,v)); au=bench(lambda:torch.ops.npu.engram_gate_w_reduce(gw_n,wh_n,we_n,x_n,k_n,v_n))
                r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
            else:
                r.append({"label":nm,"error":f"Unexpected data len {{len(data)}}","status":"FAIL"})
        except Exception as e:
            r.append({"label":nm,"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_gate_w_reduce","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"engram_gate_w_reduce","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
