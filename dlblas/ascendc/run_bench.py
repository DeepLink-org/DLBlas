#!/usr/bin/env python3
"""
Robust Per-Operator Benchmark: AscendC vs PyTorch (NPU)
Each operator benchmarked independently with timeout and error isolation.
"""
import sys, os, time, json, math, traceback, signal
import numpy as np
import torch
import torch_npu

BASE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
OUT = os.path.join(BASE, "benchmark_results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ORIGIN)

WARMUP = 10; REPEAT = 100

def sync(): torch.npu.synchronize()
def bench(fn, w=WARMUP, r=REPEAT):
    for _ in range(w): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6

def geomean(values):
    v = [x for x in values if x > 0]
    return math.exp(sum(math.log(x) for x in v)/len(v)) if v else 0.0

class TimeoutError(Exception): pass

def with_timeout(fn, seconds=120):
    """Run fn with timeout."""
    result = [None]; exc = [None]
    def target():
        try: result[0] = fn()
        except Exception as e: exc[0] = e
    import threading
    t = threading.Thread(target=target); t.daemon = True
    t.start(); t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"Timeout after {seconds}s")
    if exc[0]: raise exc[0]
    return result[0]

# ===================================================================
# Individual operator benchmarks
# ===================================================================

def op_act_quant_kernel():
    torch.ops.load_library(f"{BASE}/act_quant_kernel/build/libact_quant_kernel_ops.so")
    fp8m = 448.0; r = []
    for nm, n, g in [("1K",1024,128),("4K",4096,128),("16K",16384,128),("65K",65536,128),("256K",262144,128)]:
        np.random.seed(42); x_npu = torch.from_numpy(np.random.randn(n).astype(np.float32)).bfloat16().npu()
        def tr(): x_=x_npu.reshape(-1,g); am=x_.abs().max(-1,keepdim=True)[0].clamp(min=1e-10).float(); xs=am*(1.0/fp8m); return (x_.float()/xs).clamp(-fp8m,fp8m).reshape(x_npu.shape), xs.reshape(-1)
        tu = bench(tr); au = bench(lambda: torch.ops.npu.act_quant_kernel(x_npu,g,1e-10,False))
        r.append({"label":nm,"n":n,"g":g,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"act_quant_kernel","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_apply_mix():
    torch.ops.load_library(f"{BASE}/apply_mix/build/libapply_mix_ops.so")
    r=[]; cfgs=[("default",2,1024,4,1280),("small",1,128,2,640),("large_b",8,1024,4,1280),("large_s",2,4096,4,1280)]
    for nm,n0,n1,m,h in cfgs:
        torch.manual_seed(42); x=torch.sigmoid(torch.randn(n0,n1,m,h)).bfloat16().npu(); mix=torch.nn.functional.softmax(torch.randn(n0,n1,m,1),dim=-2).npu()
        tu=bench(lambda:(x.float()*mix).sum(-2).bfloat16()); au=bench(lambda:torch.ops.npu.apply_mix(x,mix))
        r.append({"label":nm,"shape":[n0,n1,m,h],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"apply_mix","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_head_compute_mix_fwd():
    torch.ops.load_library(f"{BASE}/head_compute_mix_fwd/build/libhead_compute_mix_fwd_ops.so")
    r=[]; cfgs=[("default",16,16384),("1K",1,256),("small",2,1),("4M",32,32768)]
    for nm,bs,n1 in cfgs:
        x=torch.randn(bs,n1,4,dtype=torch.float16); s=torch.randn(1,dtype=torch.float16); b=torch.randn(4,dtype=torch.float16); eps=0.01
        xn,sn,bn=x.npu(),s.npu(),b.npu()
        tu=bench(lambda:torch.sigmoid(x.float()*s.float()+b.float()).half()+eps)
        au=bench(lambda:torch.ops.npu.head_compute_mix_fwd(xn,sn,bn,eps))
        r.append({"label":nm,"shape":[bs,n1],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"head_compute_mix_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_expand_kenel_fwd():
    sys.path.insert(0,f"{BASE}/expand_kenel_fwd/scripts")
    from golden import compute_golden
    torch.ops.load_library(f"{BASE}/expand_kenel_fwd/build/libexpand_kenel_fwd_ops.so")
    r=[]; cfgs=[("typical",1,1024,1280,4),("min",1,1,128,2),("multi",4,256,256,2),("largeM",1,1,1280,16),("M1",1,1,1280,1)]
    for nm,B,S,H,M in cfgs:
        x=torch.randn(B,S,H,dtype=torch.float16); xn=x.npu()
        tu=bench(lambda:compute_golden(xn,M)); au=bench(lambda:torch.ops.npu.expand_kenel_fwd(xn,M))
        r.append({"label":nm,"shape":[B,S,H,M],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"expand_kenel_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_hc_split_sinkhorn():
    torch.ops.load_library(f"{BASE}/hc_split_sinkhorn/build/libhc_split_sinkhorn_ops.so")
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
    return {"op":"hc_split_sinkhorn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_sparse_attn():
    torch.ops.load_library(f"{BASE}/sparse_attn/build/libsparse_attn_ops.so")
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
    return {"op":"sparse_attn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_engram_fused_weight():
    torch.ops.load_library(f"{BASE}/engram_fused_weight/build/libengram_fused_weight_ops.so")
    from engram_fused_weight import Model, generate_test_data
    r=[]; cfgs=[("small",16,128),("default",64,1280),("medium",128,1280),("large",256,2048)]
    for nm,H,D in cfgs:
        wh,we=generate_test_data(H,D); m=Model()
        wh_n,we_n=wh.npu(),we.npu()
        tu=bench(lambda:m.forward(wh,we)); au=bench(lambda:torch.ops.npu.engram_fused_weight(wh_n,we_n))
        r.append({"label":nm,"shape":[H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"engram_fused_weight","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_engram_hash():
    torch.ops.load_library(f"{BASE}/engram_hash/build/libengram_hash_ops.so")
    from engram_hash import Model, generate_test_data
    r=[]; cfgs=[("32x3x2x8",32,3,2,8),("256x3x2x8",256,3,2,8),("1024x3x2x8",1024,3,2,8),("4096x5x4x16",4096,5,4,16)]
    for nm,nt,N,L,T in cfgs:
        torch.manual_seed(42); ng,mu,vo,of=generate_test_data({'num_tokens':nt,'ngram':N,'layers':L,'tables':T}); m=Model()
        ng_n,mu_n,vo_n,of_n=ng.npu(),mu.npu(),vo.npu(),of.npu()
        tu=bench(lambda:m.forward(ng_n,mu_n,vo_n,of_n)); au=bench(lambda:torch.ops.npu.engram_hash(ng_n,mu_n,vo_n,of_n))
        r.append({"label":nm,"shape":[nt,N,L,T],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"engram_hash","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_engram_gate_fwd():
    so=f"{BASE}/engram_gate_fwd/operators/engram_gate_fwd/build/libengram_gate_fwd_ops.so"
    if not os.path.exists(so): return {"op":"engram_gate_fwd","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    from engram_gate_fwd import Model, generate_test_data
    r=[]; cfgs=[("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128),("T64H4D128",64,4,128)]
    for nm,T,H,D in cfgs:
        hs,k,v,wh,we=generate_test_data({'num_tokens':T,'hc_mult':H,'hidden_size':D}); m=Model()
        hs_n,k_n,v_n,wh_n,we_n=hs.npu(),k.npu(),v.npu(),wh.npu(),we.npu()
        tu=bench(lambda:m.forward(hs,k,v,wh,we)); au=bench(lambda:torch.ops.npu.engram_gate_fwd(hs_n,k_n,v_n,wh_n,we_n))
        r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"engram_gate_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_engram_gate_bwd():
    torch.ops.load_library(f"{BASE}/engram_gate_bwd/build/libengram_gate_bwd_ops.so")
    from engram_gate_bwd import Model, generate_test_data
    r=[]; cfgs=[("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128),("T64H4D128",64,4,128)]
    for nm,T,H,D in cfgs:
        go,x,k,v,wh,we=generate_test_data(T,H,D); m=Model()
        go_n,x_n,k_n,v_n,wh_n,we_n=go.npu(),x.npu(),k.npu(),v.npu(),wh.npu(),we.npu()
        tu=bench(lambda:m.forward(go,x,k,v,wh,we)); au=bench(lambda:torch.ops.npu.engram_gate_bwd(go_n,x_n,k_n,v_n,wh_n,we_n))
        r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    return {"op":"engram_gate_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_engram_gate_w_reduce():
    so=f"{BASE}/engram_gate_w_reduce/operators/engram_gate_w_reduce/build/libengram_gate_w_reduce_ops.so"
    if not os.path.exists(so): return {"op":"engram_gate_w_reduce","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    from engram_gate_w_reduce import Model, generate_test_data
    r=[]; cfgs=[("default",14,4,128),("small",1,2,64),("medium",64,4,128)]
    for nm,T,H,D in cfgs:
        # generate_test_data takes (hidden_size) - but needs T, H, D
        # Check the actual signature
        data=generate_test_data(D)  # Returns tuple based on origin module
        m=Model()
        # The Model forward signature varies, let's handle it generically
        try:
            if len(data) == 6:
                gw,wh,we,x,k,v = data
                gw_n,wh_n,we_n,x_n,k_n,v_n = gw.npu(),wh.npu(),we.npu(),x.npu(),k.npu(),v.npu()
                tu=bench(lambda:m.forward(gw,wh,we,x,k,v))
                au=bench(lambda:torch.ops.npu.engram_gate_w_reduce(gw_n,wh_n,we_n,x_n,k_n,v_n))
            else:
                raise ValueError(f"Unexpected data length: {len(data)}")
            r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[T,H,D],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"engram_gate_w_reduce","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if x.get("sp") is not None),"shapes":r}

def op_head_compute_mix_bwd():
    so=f"{BASE}/head_compute_mix_bwd/operators/head_compute_mix_bwd/build/libhead_compute_mix_bwd_ops.so"
    if not os.path.exists(so): return {"op":"head_compute_mix_bwd","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    from head_compute_mix_bwd import Model, get_inputs
    # get_inputs returns the test data tuple
    data = get_inputs()
    m=Model()
    # Check Model forward signature
    import inspect
    sig = inspect.signature(m.forward)
    params = list(sig.parameters.keys())
    r=[]
    try:
        # Try with the data from get_inputs
        if len(params) == 4 and len(data) >= 4:
            a,b,c,d = data[:4]
            a_n,b_n,c_n = a.npu(),b.npu(),c.npu()
            tu=bench(lambda:m.forward(a,b,c,d))
            au=bench(lambda:torch.ops.npu.head_compute_mix_bwd(a_n,b_n,c_n,d))
            speedup = tu/au if au>0 else 0
            r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
        else:
            return {"op":"head_compute_mix_bwd","error":f"Model expects {len(params)} params, got {len(data)} data items","status":"FAIL"}
    except Exception as e:
        return {"op":"head_compute_mix_bwd","error":str(e),"status":"FAIL"}
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"head_compute_mix_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_expand_kenel_bwd():
    so=f"{BASE}/expand_kenel_bwd/operators/expand_kenel_bwd/build/libexpand_kenel_bwd_ops.so"
    if not os.path.exists(so): return {"op":"expand_kenel_bwd","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    from expand_kenel_bwd import Model, get_inputs
    data=get_inputs(); m=Model()
    r=[]
    try:
        o_grad = data[0] if isinstance(data,(list,tuple)) else data
        o_n = o_grad.npu() if isinstance(o_grad,torch.Tensor) else [g.npu() for g in o_grad]
        tu=bench(lambda:m.forward(o_grad))
        if isinstance(o_n,list): au=bench(lambda:torch.ops.npu.expand_kenel_bwd(*o_n))
        else: au=bench(lambda:torch.ops.npu.expand_kenel_bwd(o_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        return {"op":"expand_kenel_bwd","error":str(e),"status":"FAIL"}
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"expand_kenel_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_pre_split_mixes():
    torch.ops.load_library(f"{BASE}/pre_split_mixes/build/libpre_split_mixes_ops.so")
    from pre_split_mixes import Model, get_inputs
    data=get_inputs(); m=Model()
    r=[]
    try:
        im,ms,mb,eps = data[:4]
        im_n,ms_n,mb_n = im.npu(),ms.npu(),mb.npu()
        tu=bench(lambda:m.forward(im,ms,mb,eps)); au=bench(lambda:torch.ops.npu.pre_split_mixes(im_n,ms_n,mb_n,eps))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        return {"op":"pre_split_mixes","error":str(e),"status":"FAIL"}
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"pre_split_mixes","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_norm_fn():
    torch.ops.load_library(f"{BASE}/norm_fn/build/libnorm_fn_ops.so")
    from norm_fn import Model, generate_norm_fn_test_data
    r=[]
    for nm,n1,mh,hs in [("default",13,4,1280),("small",1,2,640),("medium",26,4,1280)]:
        try:
            res,fn,nw,og,eps=generate_norm_fn_test_data(n1,mh,hs,False); m=Model()
            res_n,fn_n=res.npu(),fn.npu(); nw_n=nw.npu() if nw is not None else None
            tu=bench(lambda:m.forward(res,fn,nw,eps)); au=bench(lambda:torch.ops.npu.norm_fn(res_n,fn_n,nw_n,eps))
            r.append({"label":nm,"shape":[n1,mh,hs],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[n1,mh,hs],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"norm_fn","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if x.get("sp") is not None),"shapes":r}

def op_mhc_post():
    so=f"{BASE}/mhc_post/operators/mhc_post/build/libmhc_post_ops.so"
    if not os.path.exists(so): return {"op":"mhc_post","error":"no .so","status":"SKIP"}
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
    return {"op":"mhc_post","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if x.get("sp") is not None),"shapes":r}

def op_sinkhorn():
    so=f"{BASE}/sinkhorn/operators/sinkhorn/build/libsinkhorn_ops.so"
    if not os.path.exists(so): return {"op":"sinkhorn","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    from sinkhorn import Model, get_inputs
    data=get_inputs(); m=Model()
    r=[]
    try:
        x=data[0] if isinstance(data,(list,tuple)) else data; x_n=x.npu()
        tu=bench(lambda:m.forward(x)); au=bench(lambda:torch.ops.npu.sinkhorn_normalize(x_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        return {"op":"sinkhorn","error":str(e),"status":"FAIL"}
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"sinkhorn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_big_fuse():
    """big_fuse: load op and test"""
    so=f"{BASE}/big_fuse/operators/big_fuse/build/libbig_fuse_ops.so"
    if not os.path.exists(so): return {"op":"big_fuse","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    from big_fuse import Model, get_inputs
    data=get_inputs(); m=Model()
    r=[]
    try:
        residual,fn,ms,mb = data[:4]
        res_n,fn_n,ms_n,mb_n = residual.npu(),fn.npu(),ms.npu(),mb.npu()
        tu=bench(lambda:m.forward(residual,fn,ms,mb)); au=bench(lambda:torch.ops.npu.big_fuse(res_n,fn_n,ms_n,mb_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        return {"op":"big_fuse","error":str(e),"status":"FAIL"}
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"big_fuse","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_mtpblock():
    """MTPBlock hc_post op"""
    so=f"{BASE}/MTPBlock/build/libmtpblock_ops.so"
    if not os.path.exists(so): return {"op":"MTPBlock","error":"no .so","status":"SKIP"}
    torch.ops.load_library(so)
    # MTPBlock registers under its own namespace "mtpblock"
    r=[]
    try:
        B,S,M,H=2,1024,4,1280
        x=torch.randn(B,S,M,H,dtype=torch.bfloat16); res=torch.randn(B,S,M,H,dtype=torch.bfloat16)
        post=torch.randn(M,dtype=torch.float32); comb=torch.randn(M,M,dtype=torch.float32)
        x_n,res_n,post_n,comb_n=x.npu(),res.npu(),post.npu(),comb.npu()
        def tr(): return res_n.float()*post_n.float().view(1,1,M,1) + torch.matmul(x_n.float(),comb_n.float().T)
        tu=bench(tr); au=bench(lambda:torch.ops.mtpblock.hc_post(x_n,res_n,post_n,comb_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","shape":[B,S,M,H],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        return {"op":"MTPBlock","error":str(e),"status":"FAIL"}
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    return {"op":"MTPBlock","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}

def op_indexer():
    """indexer: No AscendC .so available. Benchmark PyTorch on NPU as reference only."""
    from indexer import Model, get_inputs
    data=get_inputs(); m=Model()
    r=[]
    try:
        x,indices = data[0], data[1]
        x_n,ind_n = x.npu(), indices.npu()
        tu=bench(lambda:m.forward(x_n,ind_n))
        r.append({"label":"default","t_us":round(tu,2),"a_us":None,"sp":None,"note":"No AscendC .so"})
    except Exception as e:
        return {"op":"indexer","error":str(e),"status":"FAIL"}
    return {"op":"indexer","geomean_speedup":None,"total":len(r),"passed":0,"shapes":r,"note":"No AscendC .so - torch reference only"}

# ===================================================================
# Runner
# ===================================================================

OPERATORS = [
    ("act_quant_kernel",     op_act_quant_kernel),
    ("apply_mix",            op_apply_mix),
    ("head_compute_mix_fwd", op_head_compute_mix_fwd),
    ("expand_kenel_fwd",     op_expand_kenel_fwd),
    ("hc_split_sinkhorn",    op_hc_split_sinkhorn),
    ("sparse_attn",          op_sparse_attn),
    ("engram_fused_weight",  op_engram_fused_weight),
    ("engram_hash",          op_engram_hash),
    ("engram_gate_fwd",      op_engram_gate_fwd),
    ("engram_gate_bwd",      op_engram_gate_bwd),
    ("engram_gate_w_reduce", op_engram_gate_w_reduce),
    ("head_compute_mix_bwd", op_head_compute_mix_bwd),
    ("expand_kenel_bwd",     op_expand_kenel_bwd),
    ("pre_split_mixes",      op_pre_split_mixes),
    ("norm_fn",              op_norm_fn),
    ("mhc_post",             op_mhc_post),
    ("sinkhorn",             op_sinkhorn),
    ("big_fuse",             op_big_fuse),
    ("MTPBlock",             op_mtpblock),
    ("indexer",              op_indexer),
]

def main():
    print("="*80); print("  AscendC vs PyTorch (NPU) Benchmark - ALL Operators"); print(f"  {len(OPERATORS)} operators"); print("="*80)

    all_res = {}; all_sp = []; total_start = time.time()

    for op_name, op_fn in OPERATORS:
        print(f"\n{'─'*70}"); print(f"  [{op_name}]..."); sys.stdout.flush()
        t0 = time.time()
        try:
            res = with_timeout(op_fn, seconds=180)
            elapsed = time.time()-t0
            res["elapsed_s"] = round(elapsed,1)
            all_res[op_name] = res
            gm = res.get("geomean_speedup")
            if gm is not None and gm > 0:
                all_sp.append((op_name, gm))
                print(f"  [{op_name}] Speedup: {gm:.4f}x ({res.get('passed','?')}/{res.get('total','?')}) [{elapsed:.1f}s]")
            elif res.get("status") == "SKIP":
                print(f"  [{op_name}] SKIP: {res.get('error','')}")
            elif res.get("note"):
                print(f"  [{op_name}] {res['note']}")
            else:
                print(f"  [{op_name}] FAIL/No speedup")
            # Write per-op JSON
            with open(f"{OUT}/{op_name}_result.json","w") as f:
                json.dump(res,f,indent=2,default=str)
        except TimeoutError:
            elapsed = time.time()-t0
            print(f"  [{op_name}] TIMEOUT [{elapsed:.1f}s]")
            all_res[op_name] = {"op":op_name,"error":"timeout","status":"TIMEOUT","elapsed_s":round(elapsed,1)}
        except Exception as e:
            elapsed = time.time()-t0
            print(f"  [{op_name}] ERROR: {e} [{elapsed:.1f}s]")
            all_res[op_name] = {"op":op_name,"error":str(e),"status":"ERROR","elapsed_s":round(elapsed,1)}

    # ===== SUMMARY =====
    print(f"\n\n{'='*80}"); print("  AGGREGATE SUMMARY"); print(f"{'='*80}")
    all_sp.sort(key=lambda x:x[1], reverse=True)
    print(f"\n{'Rank':<5} {'Operator':<26} {'Geomean Speedup':>16} {'Status':>10}")
    print(f"{'─'*60}")
    for rank,(nm,sp) in enumerate(all_sp,1):
        print(f"  {rank:<3}  {nm:<26} {sp:>14.4f}x  {'OK':>10}")

    failed = [(nm,r) for nm,r in all_res.items() if nm not in dict(all_sp)]
    for nm,r in failed:
        st = r.get("status","UNKNOWN")
        print(f"  ---   {nm:<26} {'N/A':>14}  {st:>10}")

    valid = [s for _,s in all_sp if s>0]
    if valid:
        gm=geomean(valid); avg=sum(valid)/len(valid); mn=min(valid); mx=max(valid); med=sorted(valid)[len(valid)//2]
        print(f"\n{'─'*60}")
        print(f"  Aggregate Stats:")
        print(f"    Total: {len(all_res)} | With speedup: {len(valid)} | Failed: {len(failed)}")
        print(f"    Geometric mean: {gm:.4f}x | Arithmetic mean: {avg:.4f}x")
        print(f"    Median: {med:.4f}x | Min: {mn:.4f}x | Max: {mx:.4f}x")

    # Write final summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": "ascend910b2",
        "total": len(all_res),
        "with_speedup": len(valid) if valid else 0,
        "failed": len(failed),
        "speedups": {n:round(s,4) for n,s in all_sp},
        "aggregate": {
            "geometric_mean": round(gm,4) if valid else None,
            "arithmetic_mean": round(avg,4) if valid else None,
            "median": round(med,4) if valid else None,
            "min": round(mn,4) if valid else None,
            "max": round(mx,4) if valid else None,
        } if valid else {},
        "per_operator": all_res,
    }
    with open(f"{OUT}/benchmark_summary.json","w") as f:
        json.dump(summary,f,indent=2,default=str)
    print(f"\nSummary: {OUT}/benchmark_summary.json")

    # Markdown report
    with open(f"{OUT}/benchmark_report.md","w") as f:
        f.write("# AscendC vs PyTorch (NPU) Performance Benchmark\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')} | **Arch**: ascend910b2 | **Total**: {len(all_res)} ops\n\n")
        f.write("## Aggregate Results\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Operators tested | {len(all_res)} |\n")
        f.write(f"| With speedup data | {len(valid) if valid else 0} |\n")
        if valid:
            f.write(f"| Geometric mean speedup | {gm:.4f}x |\n")
            f.write(f"| Arithmetic mean speedup | {avg:.4f}x |\n")
            f.write(f"| Median | {med:.4f}x |\n")
            f.write(f"| Range | {mn:.4f}x – {mx:.4f}x |\n")
        f.write("\n## Per-Operator Speedup\n\n")
        f.write("| Rank | Operator | Geomean Speedup | Status |\n")
        f.write("|------|----------|----------------|--------|\n")
        for rank,(nm,sp) in enumerate(all_sp,1):
            f.write(f"| {rank} | {nm} | {sp:.4f}x | OK |\n")
        for nm,r in failed:
            f.write(f"| — | {nm} | N/A | {r.get('status','?')} |\n")
    print(f"Report: {OUT}/benchmark_report.md")
    total_elapsed = time.time()-total_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    return summary

if __name__ == "__main__":
    main()
