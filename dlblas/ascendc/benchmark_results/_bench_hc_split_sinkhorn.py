#!/usr/bin/env python3
# Auto-generated benchmark for hc_split_sinkhorn
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

    torch.ops.load_library(f"/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/hc_split_sinkhorn/build/libhc_split_sinkhorn_ops.so")
    r=[]; cfgs=[("b2s8hc4",2,8,4,20,1e-6),("b1s1hc4",1,1,4,20,1e-6),("b64s8hc4",64,8,4,20,1e-6),("b4s16hc4",4,16,4,20,1e-6),("b8s4hc8",8,4,8,20,1e-6)]
    for nm,b,s,hc,iters,eps in cfgs:
        mc=(2+hc)*hc; np.random.seed(0)
        mn=torch.from_numpy(np.random.randn(b,s,mc).astype(np.float32)).npu(); sc=torch.tensor([0.5,0.25,1.0],dtype=torch.float32).npu()
        bs_=torch.from_numpy(np.random.randn(mc).astype(np.float32)*0.1).npu()
        pr=torch.empty(b,s,hc,dtype=torch.float32,device='npu'); po=torch.empty(b,s,hc,dtype=torch.float32,device='npu'); cb=torch.empty(b,s,hc,hc,dtype=torch.float32,device='npu')
        def tr():
            B_=b*s; x=mn.reshape(-1,mc); s0,s1,s2=sc[0],sc[1],sc[2]
            pre=torch.sigmoid(x[:,:hc]*s0+bs_[:hc].unsqueeze(0))+eps
            post=2*torch.sigmoid(x[:,hc:2*hc]*s1+bs_[hc:2*hc].unsqueeze(0))
            raw=x[:,2*hc:2*hc+hc*hc]; comb=raw.view(-1,hc,hc)*s2+bs_[2*hc:2*hc+hc*hc].view(1,hc,hc)
            rm=comb.amax(-1,keepdim=True); comb=torch.exp(comb-rm); comb=comb/comb.sum(-1,keepdim=True)+eps
            comb=comb/(comb.sum(-2,keepdim=True)+eps)
            for _ in range(iters-1): comb=comb/(comb.sum(-1,keepdim=True)+eps); comb=comb/(comb.sum(-2,keepdim=True)+eps)
        tu=bench(tr); au=bench(lambda:torch.ops.npu.hc_split_sinkhorn(mn,hc,iters,eps,sc,bs_,pr,po,cb))
        r.append({"label":nm,"shape":[b,s,hc],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"hc_split_sinkhorn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))

except Exception as e:
    import traceback
    print(json.dumps({"op":"hc_split_sinkhorn","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}))
    sys.exit(0)
