#!/usr/bin/env python3
# Auto-generated benchmark for MTPBlock
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

    so=f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/MTPBlock/build/libmtpblock_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"MTPBlock","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    r=[]
    try:
        B,S,M,H=2,1024,4,1280
        x=torch.randn(B,S,M,H,dtype=torch.bfloat16); res=torch.randn(B,S,M,H,dtype=torch.bfloat16)
        post=torch.randn(M,dtype=torch.float32); comb=torch.randn(M,M,dtype=torch.float32)
        x_n,res_n,post_n,comb_n=x.npu(),res.npu(),post.npu(),comb.npu()
        def tr(): return res_n.float()*post_n.float().view(1,1,M,1)+torch.matmul(x_n.float(),comb_n.float().T)
        tu=bench(tr); au=bench(lambda:torch.ops.mtpblock.hc_post(x_n,res_n,post_n,comb_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","shape":[B,S,M,H],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"MTPBlock","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"MTPBlock","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
