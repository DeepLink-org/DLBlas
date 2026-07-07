#!/usr/bin/env python3
# Auto-generated benchmark for mhc_post
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

    so=f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/mhc_post/operators/mhc_post/build/libmhc_post_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"mhc_post","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from mhc_post import Model, generate_mhc_post_test_data
    r=[]
    for nm,B,S,M,H in [("default",1,1024,4,1280),("small",1,64,2,640),("medium",4,512,4,1280)]:
        try:
            x,res,plm,comb=generate_mhc_post_test_data(B,S,M,H); m=Model()
            x_n,res_n,plm_n,comb_n=x.npu(),res.npu(),plm.npu(),comb.npu()
            tu=bench(lambda:m.forward(x,res,plm,comb)); au=bench(lambda:torch.ops.npu.mhc_post(x_n,res_n,plm_n,comb_n))
            r.append({"label":nm,"shape":[B,S,M,H],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[B,S,M,H],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"mhc_post","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"mhc_post","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
