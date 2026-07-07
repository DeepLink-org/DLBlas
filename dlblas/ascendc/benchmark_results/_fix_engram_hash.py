
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from engram_hash import Model, generate_test_data

torch.ops.load_library("/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_hash/build/libengram_hash_ops.so")

WARMUP=10; REPEAT=100
def sync(): torch.npu.synchronize()
def bench(fn,w=WARMUP,r=REPEAT):
    for _ in range(w): fn()
    sync(); t0=time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6

def geomean(v):
    vv=[x for x in v if x>0]
    return math.exp(sum(math.log(x) for x in vv)/len(vv)) if vv else 0

r=[]
configs = [("32x3x2x8",32,3,2,8),("256x3x2x8",256,3,2,8),("1024x3x2x8",1024,3,2,8),("4096x5x4x16",4096,5,4,16)]
for nm,nt,N,L,T in configs:
    try:
        torch.manual_seed(42)
        params = dict(num_tokens=nt, ngram=N, layers=L, tables=T)
        ng,mu,vo,of=generate_test_data(params); m=Model()
        ng_n,mu_n,vo_n,of_n=ng.npu(),mu.npu(),vo.npu(),of.npu()
        tu=bench(lambda:m.forward(ng_n,mu_n,vo_n,of_n)); au=bench(lambda:torch.ops.npu.engram_hash(ng_n,mu_n,vo_n,of_n))
        r.append({"label":nm,"shape":[nt,N,L,T],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    except Exception as e:
        r.append({"label":nm,"shape":[nt,N,L,T],"error":str(e),"status":"FAIL"})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"engram_hash","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
