
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")

so="/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/MTPBlock/build/libmtpblock_ops.so"
if not os.path.exists(so):
    print(json.dumps({"op":"MTPBlock","error":"no .so","status":"SKIP"}))
    sys.exit(0)
torch.ops.load_library(so)

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

# The MTPBlock's hc_post uses torch.ops.mtpblock.hc_post (separate namespace)
import inspect
# Check what ops are available
ops = [op for op in dir(torch.ops.mtpblock) if not op.startswith('_')]
print(f"mtpblock ops: {ops}", file=sys.stderr)

r=[]
try:
    B,S,M,H=2,1024,4,1280
    x=torch.randn(B,S,M,H,dtype=torch.bfloat16); res=torch.randn(B,S,M,H,dtype=torch.bfloat16)
    post=torch.randn(M,dtype=torch.float32); comb=torch.randn(M,M,dtype=torch.float32)
    x_n,res_n,post_n,comb_n=x.npu(),res.npu(),post.npu(),comb.npu()
    # torch reference: res + matmul(x, comb.T) * post
    def tr(): return res_n.float()+torch.matmul(x_n.float(),comb_n.float().T)*post_n.float().view(1,1,M,1)
    tu=bench(tr); au=bench(lambda:torch.ops.mtpblock.hc_post(x_n,res_n,post_n,comb_n))
    speedup=tu/au if au>0 else 0
    r=[{"label":"default","shape":[B,S,M,H],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}]
except Exception as e:
    import traceback
    r=[{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"MTPBlock","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
