#!/usr/bin/env python3
# Auto-generated benchmark for indexer
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

    from indexer import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        x,indices=data[0],data[1]; x_n,ind_n=x.npu(),indices.npu()
        tu=bench(lambda:m.forward(x_n,ind_n))
        r.append({"label":"default","t_us":round(tu,2),"a_us":None,"sp":None,"note":"No AscendC .so"})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    print(json.dumps({"op":"indexer","geomean_speedup":None,"total":len(r),"passed":0,"shapes":r,"note":"No AscendC .so available"}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"indexer","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
