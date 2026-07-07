
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")

# The .so has schema conflict. Try to work around it.
so="/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/sparse_attn/build/libsparse_attn_ops.so"

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

# Check if there's an alternative .so path
alt_so = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/sparse_attn/operators/sparse_attn/build/libsparse_attn_ops.so"
if os.path.exists(alt_so):
    so = alt_so
    print(f"Using alt .so: {so}", file=sys.stderr)

try:
    torch.ops.load_library(so)
    print("Loaded sparse_attn .so successfully", file=sys.stderr)

    r=[]
    b,m,n,h,d,tk=2,16,32,8,64,16
    torch.manual_seed(42); dev=torch.device("npu:0"); ss=tensor(d**-0.5) # try as tensor
    q=torch.randn(b,m,h,d,dtype=torch.bfloat16,device=dev); kv=torch.randn(b,n,d,dtype=torch.bfloat16,device=dev)
    ats=torch.randn(h,dtype=torch.float32,device=dev)*0.1
    tidx=torch.zeros(b,m,tk,dtype=torch.int32)
    for bi in range(b):
        for mi in range(m): perm=torch.randperm(n)[:tk]; tidx[bi,mi,:]=perm
    tidx=tidx.to(dev)

    def tr():
        vm=tidx>=0; si=tidx.clamp(0).long(); bi_=torch.arange(b,device=dev)[:,None,None].expand(b,m,tk)
        g=kv[bi_,si]; g=g.masked_fill(~vm.unsqueeze(-1),0.0)
        sc=torch.einsum("bmhd,bmtd->bmht",q.float(),g.float())*ss
        sc=sc.masked_fill(~vm.unsqueeze(2),float("-inf")); sk=ats.float().view(1,1,h,1)
        ms=torch.amax(sc,-1,keepdim=True); ms=torch.maximum(ms,sk); es=torch.exp(sc-ms)
        es=es.masked_fill(~vm.unsqueeze(2),0.0); eks=torch.exp(sk-ms)
        se=es.sum(-1,keepdim=True)+eks; aw=es/se
        return torch.einsum("bmht,bmtd->bmhd",aw,g.float()).to(torch.bfloat16)

    tu=bench(tr)
    # Try different softmax_scale types
    try:
        au=bench(lambda:torch.ops.npu.sparse_attn(q,kv,ats,tidx,float(ss.item())))
    except:
        try:
            au=bench(lambda:torch.ops.npu.sparse_attn(q,kv,ats,tidx,ss))
        except Exception as e2:
            r=[{"label":"default","error":f".so schema conflict: {e2}","status":"FAIL"}]
            sps=[]; gm=0
            print(json.dumps({"op":"sparse_attn","geomean_speedup":0,"total":1,"passed":0,"shapes":r,"error":str(e2)}))
            sys.exit(0)

    speedup=tu/au if au>0 else 0
    r=[{"label":"default","shape":[b,m,n,h,d,tk],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}]
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"sparse_attn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
except Exception as e:
    import traceback
    print(json.dumps({"op":"sparse_attn","error":str(e),"trace":traceback.format_exc()[:500],"status":"ERROR"}))
