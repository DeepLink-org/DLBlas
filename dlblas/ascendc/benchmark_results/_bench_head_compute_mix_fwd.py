#!/usr/bin/env python3
# Auto-generated benchmark for head_compute_mix_fwd
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/head_compute_mix_fwd/build/libhead_compute_mix_fwd_ops.so")
    r=[]; cfgs=[("default",16,16384),("1K",1,256),("small",2,1),("4M",32,32768)]
    for nm,bs,n1 in cfgs:
        x=torch.randn(bs,n1,4,dtype=torch.float16); s=torch.randn(1,dtype=torch.float16); b=torch.randn(4,dtype=torch.float16); eps=0.01
        xn,sn,bn=x.npu(),s.npu(),b.npu()
        tu=bench(lambda:torch.sigmoid(x.float()*s.float()+b.float()).half()+eps)
        au=bench(lambda:torch.ops.npu.head_compute_mix_fwd(xn,sn,bn,eps))
        r.append({"label":nm,"shape":[bs,n1],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"head_compute_mix_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"head_compute_mix_fwd","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
