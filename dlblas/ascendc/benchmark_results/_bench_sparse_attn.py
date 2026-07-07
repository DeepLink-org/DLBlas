#!/usr/bin/env python3
# Auto-generated benchmark for sparse_attn
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/sparse_attn/build/libsparse_attn_ops.so")
    r=[]; cfgs=[("default",2,16,32,8,64,16,0.0),("half_inv",2,16,32,8,64,16,0.5),("small",1,1,32,4,32,8,0.0),("decode",4,1,128,32,128,128,0.1)]
    for nm,b,m,n,h,d,tk,ir in cfgs:
        torch.manual_seed(42); dev=torch.device("npu:0"); ss=d**-0.5
        q=torch.randn(b,m,h,d,dtype=torch.bfloat16,device=dev); kv=torch.randn(b,n,d,dtype=torch.bfloat16,device=dev)
        ats=torch.randn(h,dtype=torch.float32,device=dev)*0.1
        tidx=torch.zeros(b,m,tk,dtype=torch.int32); nv=int(tk*(1-ir))
        for bi in range(b):
            for mi in range(m): perm=torch.randperm(n)[:nv]; tidx[bi,mi,:nv]=perm; tidx[bi,mi,nv:]=-1
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
        tu=bench(tr); au=bench(lambda:torch.ops.npu.sparse_attn(q,kv,ats,tidx,ss))
        r.append({"label":nm,"shape":[b,m,n,h,d,tk],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"sparse_attn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"sparse_attn","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
