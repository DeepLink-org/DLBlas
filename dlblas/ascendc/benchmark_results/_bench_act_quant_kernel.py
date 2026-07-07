#!/usr/bin/env python3
# Auto-generated benchmark for act_quant_kernel
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/act_quant_kernel/build/libact_quant_kernel_ops.so")
    fp8m=448.0; r=[]
    for nm,n,g in [("1K",1024,128),("4K",4096,128),("16K",16384,128),("65K",65536,128),("256K",262144,128)]:
        np.random.seed(42); x_npu=torch.from_numpy(np.random.randn(n).astype(np.float32)).bfloat16().npu()
        def tr(): x_=x_npu.reshape(-1,g); am=x_.abs().max(-1,keepdim=True)[0].clamp(min=1e-10).float(); xs=am*(1.0/fp8m); return (x_.float()/xs).clamp(-fp8m,fp8m).reshape(x_npu.shape),xs.reshape(-1)
        tu=bench(tr); au=bench(lambda:torch.ops.npu.act_quant_kernel(x_npu,g,1e-10,False))
        r.append({"label":nm,"n":n,"g":g,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"act_quant_kernel","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"act_quant_kernel","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
