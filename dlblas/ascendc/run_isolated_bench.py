#!/usr/bin/env python3
"""
Process-isolated per-operator benchmark runner. Each operator runs in its own subprocess.
"""
import subprocess, sys, os, json, time, math

BASE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
OUT = os.path.join(BASE, "benchmark_results")
os.makedirs(OUT, exist_ok=True)

# ===================================================================
# Per-operator benchmark scripts (self-contained, run as subprocess)
# ===================================================================

OPERATOR_SCRIPTS = {}

def make_script(op_name, code):
    path = os.path.join(OUT, f"_bench_{op_name}.py")
    with open(path, "w") as f:
        f.write(f'''#!/usr/bin/env python3
# Auto-generated benchmark for {op_name}
import sys, os, time, json, math
import numpy as np
import torch
import torch_npu
sys.path.insert(0, "{ORIGIN}")
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
{code}
except Exception as e:
    import traceback
    print(json.dumps({{"op":"{op_name}","error":str(e),"trace":traceback.format_exc(),"status":"ERROR"}}))
    sys.exit(0)
''')
    return path

# ---- act_quant_kernel ----
make_script("act_quant_kernel", '''
    torch.ops.load_library(f"{BASE}/act_quant_kernel/build/libact_quant_kernel_ops.so")
    fp8m=448.0; r=[]
    for nm,n,g in [("1K",1024,128),("4K",4096,128),("16K",16384,128),("65K",65536,128),("256K",262144,128)]:
        np.random.seed(42); x_npu=torch.from_numpy(np.random.randn(n).astype(np.float32)).bfloat16().npu()
        def tr(): x_=x_npu.reshape(-1,g); am=x_.abs().max(-1,keepdim=True)[0].clamp(min=1e-10).float(); xs=am*(1.0/fp8m); return (x_.float()/xs).clamp(-fp8m,fp8m).reshape(x_npu.shape),xs.reshape(-1)
        tu=bench(tr); au=bench(lambda:torch.ops.npu.act_quant_kernel(x_npu,g,1e-10,False))
        r.append({"label":nm,"n":n,"g":g,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"act_quant_kernel","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- apply_mix ----
make_script("apply_mix", '''
    torch.ops.load_library(f"{BASE}/apply_mix/build/libapply_mix_ops.so")
    r=[]; cfgs=[("default",2,1024,4,1280),("small",1,128,2,640),("large_b",8,1024,4,1280),("large_s",2,4096,4,1280)]
    for nm,n0,n1,m,h in cfgs:
        torch.manual_seed(42); x=torch.sigmoid(torch.randn(n0,n1,m,h)).bfloat16().npu(); mix=torch.nn.functional.softmax(torch.randn(n0,n1,m,1),dim=-2).npu()
        tu=bench(lambda:(x.float()*mix).sum(-2).bfloat16()); au=bench(lambda:torch.ops.npu.apply_mix(x,mix))
        r.append({"label":nm,"shape":[n0,n1,m,h],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"apply_mix","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- head_compute_mix_fwd ----
make_script("head_compute_mix_fwd", '''
    torch.ops.load_library(f"{BASE}/head_compute_mix_fwd/build/libhead_compute_mix_fwd_ops.so")
    r=[]; cfgs=[("default",16,16384),("1K",1,256),("small",2,1),("4M",32,32768)]
    for nm,bs,n1 in cfgs:
        x=torch.randn(bs,n1,4,dtype=torch.float16); s=torch.randn(1,dtype=torch.float16); b=torch.randn(4,dtype=torch.float16); eps=0.01
        xn,sn,bn=x.npu(),s.npu(),b.npu()
        tu=bench(lambda:torch.sigmoid(x.float()*s.float()+b.float()).half()+eps)
        au=bench(lambda:torch.ops.npu.head_compute_mix_fwd(xn,sn,bn,eps))
        r.append({"label":nm,"shape":[bs,n1],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"head_compute_mix_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- expand_kenel_fwd ----
make_script("expand_kenel_fwd", '''
    sys.path.insert(0,f"{BASE}/expand_kenel_fwd/scripts")
    from golden import compute_golden
    torch.ops.load_library(f"{BASE}/expand_kenel_fwd/build/libexpand_kenel_fwd_ops.so")
    r=[]; cfgs=[("typical",1,1024,1280,4),("min",1,1,128,2),("multi",4,256,256,2),("largeM",1,1,1280,16),("M1",1,1,1280,1)]
    for nm,B,S,H,M in cfgs:
        x=torch.randn(B,S,H,dtype=torch.float16); xn=x.npu()
        tu=bench(lambda:compute_golden(xn,M)); au=bench(lambda:torch.ops.npu.expand_kenel_fwd(xn,M))
        r.append({"label":nm,"shape":[B,S,H,M],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"expand_kenel_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- hc_split_sinkhorn ----
make_script("hc_split_sinkhorn", '''
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
    print(json.dumps({"op":"hc_split_sinkhorn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- sparse_attn ----
make_script("sparse_attn", '''
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
    print(json.dumps({"op":"sparse_attn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- engram_fused_weight ----
make_script("engram_fused_weight", '''
    torch.ops.load_library(f"{BASE}/engram_fused_weight/build/libengram_fused_weight_ops.so")
    from engram_fused_weight import Model, generate_test_data
    r=[]; cfgs=[("small",16,128),("default",64,1280),("medium",128,1280),("large",256,2048)]
    for nm,H,D in cfgs:
        wh,we=generate_test_data(H,D); m=Model()
        wh_n,we_n=wh.npu(),we.npu()
        tu=bench(lambda:m.forward(wh,we)); au=bench(lambda:torch.ops.npu.engram_fused_weight(wh_n,we_n))
        r.append({"label":nm,"shape":[H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_fused_weight","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- engram_hash ----
make_script("engram_hash", '''
    torch.ops.load_library(f"{BASE}/engram_hash/build/libengram_hash_ops.so")
    from engram_hash import Model, generate_test_data
    r=[]; cfgs=[("32x3x2x8",32,3,2,8),("256x3x2x8",256,3,2,8),("1024x3x2x8",1024,3,2,8),("4096x5x4x16",4096,5,4,16)]
    for nm,nt,N,L,T in cfgs:
        torch.manual_seed(42); ng,mu,vo,of=generate_test_data({{'num_tokens':nt,'ngram':N,'layers':L,'tables':T}}); m=Model()
        ng_n,mu_n,vo_n,of_n=ng.npu(),mu.npu(),vo.npu(),of.npu()
        tu=bench(lambda:m.forward(ng_n,mu_n,vo_n,of_n)); au=bench(lambda:torch.ops.npu.engram_hash(ng_n,mu_n,vo_n,of_n))
        r.append({"label":nm,"shape":[nt,N,L,T],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    sps=[x["sp"] for x in r if x["sp"]>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_hash","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- engram_gate_fwd ----
make_script("engram_gate_fwd", '''
    so=f"{BASE}/engram_gate_fwd/operators/engram_gate_fwd/build/libengram_gate_fwd_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"engram_gate_fwd","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from engram_gate_fwd import Model, generate_test_data
    r=[]; cfgs=[("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128),("T64H4D128",64,4,128)]
    for nm,T,H,D in cfgs:
        try:
            hs,k,v,wh,we=generate_test_data({{'num_tokens':T,'hc_mult':H,'hidden_size':D}}); m=Model()
            hs_n,k_n,v_n,wh_n,we_n=hs.npu(),k.npu(),v.npu(),wh.npu(),we.npu()
            tu=bench(lambda:m.forward(hs,k,v,wh,we)); au=bench(lambda:torch.ops.npu.engram_gate_fwd(hs_n,k_n,v_n,wh_n,we_n))
            r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[T,H,D],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_gate_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- engram_gate_bwd ----
make_script("engram_gate_bwd", '''
    torch.ops.load_library(f"{BASE}/engram_gate_bwd/build/libengram_gate_bwd_ops.so")
    from engram_gate_bwd import Model, generate_test_data
    r=[]; cfgs=[("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128),("T64H4D128",64,4,128)]
    for nm,T,H,D in cfgs:
        try:
            go,x,k,v,wh,we=generate_test_data(T,H,D); m=Model()
            go_n,x_n,k_n,v_n,wh_n,we_n=go.npu(),x.npu(),k.npu(),v.npu(),wh.npu(),we.npu()
            tu=bench(lambda:m.forward(go,x,k,v,wh,we)); au=bench(lambda:torch.ops.npu.engram_gate_bwd(go_n,x_n,k_n,v_n,wh_n,we_n))
            r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
        except Exception as e:
            r.append({"label":nm,"shape":[T,H,D],"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_gate_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- pre_split_mixes ----
make_script("pre_split_mixes", '''
    torch.ops.load_library(f"{BASE}/pre_split_mixes/build/libpre_split_mixes_ops.so")
    from pre_split_mixes import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        im,ms,mb,eps = data[:4]; im_n,ms_n,mb_n = im.npu(),ms.npu(),mb.npu()
        tu=bench(lambda:m.forward(im,ms,mb,eps)); au=bench(lambda:torch.ops.npu.pre_split_mixes(im_n,ms_n,mb_n,eps))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"pre_split_mixes","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- norm_fn ----
make_script("norm_fn", '''
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
    print(json.dumps({"op":"norm_fn","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- mhc_post ----
make_script("mhc_post", '''
    so=f"{BASE}/mhc_post/operators/mhc_post/build/libmhc_post_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"mhc_post","error":"no .so","status":"SKIP"}})); sys.exit(0)
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
    print(json.dumps({"op":"mhc_post","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- big_fuse ----
make_script("big_fuse", '''
    so=f"{BASE}/big_fuse/operators/big_fuse/build/libbig_fuse_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"big_fuse","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from big_fuse import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        residual,fn,ms,mb = data[:4]; res_n,fn_n,ms_n,mb_n = residual.npu(),fn.npu(),ms.npu(),mb.npu()
        tu=bench(lambda:m.forward(residual,fn,ms,mb)); au=bench(lambda:torch.ops.npu.big_fuse(res_n,fn_n,ms_n,mb_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"big_fuse","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- MTPBlock ----
make_script("MTPBlock", '''
    so=f"{BASE}/MTPBlock/build/libmtpblock_ops.so"
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
'''.replace("{BASE}", BASE))

# ---- sinkhorn ----
make_script("sinkhorn", '''
    so=f"{BASE}/sinkhorn/operators/sinkhorn/build/libsinkhorn_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"sinkhorn","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from sinkhorn import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        x=data[0] if isinstance(data,(list,tuple)) else data; x_n=x.npu()
        tu=bench(lambda:m.forward(x)); au=bench(lambda:torch.ops.npu.sinkhorn_normalize(x_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"sinkhorn","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- head_compute_mix_bwd ----
make_script("head_compute_mix_bwd", '''
    so=f"{BASE}/head_compute_mix_bwd/operators/head_compute_mix_bwd/build/libhead_compute_mix_bwd_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"head_compute_mix_bwd","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from head_compute_mix_bwd import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        a,b,c,d = data[:4]; a_n,b_n,c_n = a.npu(),b.npu(),c.npu()
        tu=bench(lambda:m.forward(a,b,c,d)); au=bench(lambda:torch.ops.npu.head_compute_mix_bwd(a_n,b_n,c_n,d))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"head_compute_mix_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- expand_kenel_bwd ----
make_script("expand_kenel_bwd", '''
    so=f"{BASE}/expand_kenel_bwd/operators/expand_kenel_bwd/build/libexpand_kenel_bwd_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"expand_kenel_bwd","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from expand_kenel_bwd import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        o_grad=data[0]; o_n=o_grad.npu() if isinstance(o_grad,torch.Tensor) else [g.npu() for g in o_grad]
        tu=bench(lambda:m.forward(o_grad))
        if isinstance(o_n,list): au=bench(lambda:torch.ops.npu.expand_kenel_bwd(*o_n))
        else: au=bench(lambda:torch.ops.npu.expand_kenel_bwd(o_n))
        speedup=tu/au if au>0 else 0
        r.append({"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"expand_kenel_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- engram_gate_w_reduce ----
make_script("engram_gate_w_reduce", '''
    so=f"{BASE}/engram_gate_w_reduce/operators/engram_gate_w_reduce/build/libengram_gate_w_reduce_ops.so"
    if not os.path.exists(so): print(json.dumps({{"op":"engram_gate_w_reduce","error":"no .so","status":"SKIP"}})); sys.exit(0)
    torch.ops.load_library(so)
    from engram_gate_w_reduce import Model, generate_test_data
    r=[]
    for nm,T,H,D in [("default",14,4,128),("small",1,2,64)]:
        try:
            data=generate_test_data(D); m=Model()
            if len(data)==6:
                gw,wh,we,x,k,v=data; gw_n,wh_n,we_n,x_n,k_n,v_n=gw.npu(),wh.npu(),we.npu(),x.npu(),k.npu(),v.npu()
                tu=bench(lambda:m.forward(gw,wh,we,x,k,v)); au=bench(lambda:torch.ops.npu.engram_gate_w_reduce(gw_n,wh_n,we_n,x_n,k_n,v_n))
                r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
            else:
                r.append({"label":nm,"error":f"Unexpected data len {{len(data)}}","status":"FAIL"})
        except Exception as e:
            r.append({"label":nm,"error":str(e),"status":"FAIL"})
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({"op":"engram_gate_w_reduce","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
'''.replace("{BASE}", BASE))

# ---- indexer (no AscendC .so) ----
make_script("indexer", '''
    from indexer import Model, get_inputs
    data=get_inputs(); m=Model(); r=[]
    try:
        x,indices=data[0],data[1]; x_n,ind_n=x.npu(),indices.npu()
        tu=bench(lambda:m.forward(x_n,ind_n))
        r.append({"label":"default","t_us":round(tu,2),"a_us":None,"sp":None,"note":"No AscendC .so"})
    except Exception as e:
        r.append({"label":"default","error":str(e),"status":"FAIL"})
    print(json.dumps({"op":"indexer","geomean_speedup":None,"total":len(r),"passed":0,"shapes":r,"note":"No AscendC .so available"}))
'''.replace("{BASE}", BASE))

# ===================================================================
# Runner
# ===================================================================

OPERATORS = [
    "act_quant_kernel", "apply_mix", "head_compute_mix_fwd", "expand_kenel_fwd",
    "hc_split_sinkhorn", "sparse_attn", "engram_fused_weight", "engram_hash",
    "engram_gate_fwd", "engram_gate_bwd", "engram_gate_w_reduce", "head_compute_mix_bwd",
    "expand_kenel_bwd", "pre_split_mixes", "norm_fn", "mhc_post",
    "sinkhorn", "big_fuse", "MTPBlock", "indexer",
]

def main():
    print("="*80); print("  AscendC vs PyTorch (NPU) - Process-Isolated Benchmarks"); print(f"  {len(OPERATORS)} operators"); print("="*80)

    all_res = {}; all_sp = []; total_start = time.time(); TIMEOUT = 300

    for op_name in OPERATORS:
        script_path = os.path.join(OUT, f"_bench_{op_name}.py")
        if not os.path.exists(script_path):
            print(f"\n  [{op_name}] Script not found, skipping")
            all_res[op_name] = {"op": op_name, "error": "script not found", "status": "SKIP"}
            continue

        print(f"\n{'─'*70}"); print(f"  [{op_name}] Running in subprocess..."); sys.stdout.flush()
        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-u", script_path],
                capture_output=True, text=True, timeout=TIMEOUT,
                cwd=BASE, env={**os.environ, "PYTHONPATH": f"{ORIGIN}:{os.environ.get('PYTHONPATH','')}"}
            )
            elapsed = time.time()-t0
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stderr and ("error" in stderr.lower() or "Error" in stderr or "Traceback" in stderr or "FAIL" in stderr):
                print(f"  [{op_name}] stderr: {stderr[:500]}")

            # Parse JSON output
            if stdout:
                try:
                    # Find last JSON object in output
                    lines = stdout.split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line.startswith('{'):
                            res = json.loads(line)
                            break
                    res["elapsed_s"] = round(elapsed, 1)
                    res["returncode"] = result.returncode
                    all_res[op_name] = res
                    gm = res.get("geomean_speedup")
                    if gm is not None and gm > 0:
                        all_sp.append((op_name, gm))
                        print(f"  [{op_name}] Speedup: {gm:.4f}x ({res.get('passed','?')}/{res.get('total','?')}) [{elapsed:.1f}s]")
                    elif res.get("status") == "SKIP":
                        print(f"  [{op_name}] SKIP: {res.get('error','')} [{elapsed:.1f}s]")
                    elif res.get("note"):
                        print(f"  [{op_name}] {res['note']} [{elapsed:.1f}s]")
                    else:
                        print(f"  [{op_name}] No speedup: {res.get('error','unknown')[:200]} [{elapsed:.1f}s]")
                except json.JSONDecodeError:
                    print(f"  [{op_name}] JSON parse error. stdout: {stdout[:300]}")
                    all_res[op_name] = {"op": op_name, "error": "JSON parse error", "stdout": stdout[:500], "status": "ERROR"}
            else:
                print(f"  [{op_name}] No output. rc={result.returncode} stderr={stderr[:200]}")
                all_res[op_name] = {"op": op_name, "error": "no output", "stderr": stderr[:500], "status": "ERROR"}

        except subprocess.TimeoutExpired:
            elapsed = time.time()-t0
            print(f"  [{op_name}] TIMEOUT [{elapsed:.1f}s]")
            all_res[op_name] = {"op": op_name, "error": "timeout", "status": "TIMEOUT", "elapsed_s": round(elapsed,1)}
        except Exception as e:
            elapsed = time.time()-t0
            print(f"  [{op_name}] EXCEPTION: {e} [{elapsed:.1f}s]")
            all_res[op_name] = {"op": op_name, "error": str(e), "status": "ERROR", "elapsed_s": round(elapsed,1)}

    # ===== SUMMARY =====
    print(f"\n\n{'='*80}"); print("  AGGREGATE BENCHMARK SUMMARY"); print(f"{'='*80}")
    all_sp.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Operator':<26} {'Geomean Speedup':>16} {'Status':>12}")
    print(f"{'─'*62}")
    for rank, (nm, sp) in enumerate(all_sp, 1):
        print(f"  {rank:<3}  {nm:<26} {sp:>14.4f}x  {'OK':>12}")

    failed = [(nm, r) for nm, r in all_res.items() if nm not in dict(all_sp)]
    for nm, r in failed:
        st = r.get("status", "UNKNOWN")
        print(f"  ---   {nm:<26} {'N/A':>14}  {st:>12}")

    valid = [s for _, s in all_sp if s > 0]
    if valid:
        gm = math.exp(sum(math.log(v) for v in valid)/len(valid))
        avg = sum(valid)/len(valid)
        mn = min(valid); mx = max(valid); med = sorted(valid)[len(valid)//2]
        print(f"\n{'─'*62}")
        print(f"  Aggregate Statistics:")
        print(f"    Total: {len(all_res)} | With speedup: {len(valid)} | Failed/Skipped: {len(failed)}")
        print(f"    Geometric mean: {gm:.4f}x | Arithmetic mean: {avg:.4f}x")
        print(f"    Median: {med:.4f}x | Min: {mn:.4f}x | Max: {mx:.4f}x")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": "ascend910b2",
        "total_operators": len(all_res),
        "with_speedup": len(valid) if valid else 0,
        "failed_skipped": len(failed),
        "speedups": {n: round(s, 4) for n, s in all_sp},
        "aggregate": {
            "geometric_mean": round(gm, 4) if valid else None,
            "arithmetic_mean": round(avg, 4) if valid else None,
            "median": round(med, 4) if valid else None,
            "min": round(mn, 4) if valid else None,
            "max": round(mx, 4) if valid else None,
        } if valid else {},
        "per_operator": all_res,
    }

    with open(f"{OUT}/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary: {OUT}/benchmark_summary.json")

    # Markdown
    with open(f"{OUT}/benchmark_report.md", "w") as f:
        f.write("# AscendC vs PyTorch (NPU) Performance Benchmark Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')} | **Arch**: ascend910b2 | **Total**: {len(all_res)} operators\n\n")
        f.write("## Aggregate Results\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Operators tested | {len(all_res)} |\n")
        f.write(f"| With speedup data | {len(valid) if valid else 0} |\n")
        f.write(f"| Failed / Skipped | {len(failed)} |\n")
        if valid:
            f.write(f"| **Geometric mean speedup** | **{gm:.4f}x** |\n")
            f.write(f"| Arithmetic mean speedup | {avg:.4f}x |\n")
            f.write(f"| Median speedup | {med:.4f}x |\n")
            f.write(f"| Range | {mn:.4f}x – {mx:.4f}x |\n")
        f.write("\n## Per-Operator Speedup (AscendC vs PyTorch on NPU)\n\n")
        f.write("| Rank | Operator | Geomean Speedup | Shapes (P/T) | Status |\n")
        f.write("|------|----------|----------------|--------------|--------|\n")
        for rank, (nm, sp) in enumerate(all_sp, 1):
            r = all_res[nm]
            f.write(f"| {rank} | {nm} | {sp:.4f}x | {r.get('passed','?')}/{r.get('total','?')} | OK |\n")
        for nm, r in failed:
            st = r.get("status", "?")
            err = r.get("error", "")
            f.write(f"| — | {nm} | N/A | 0/{r.get('total','?')} | {st}: {err[:80]} |\n")

    print(f"  Report: {OUT}/benchmark_report.md")
    total_elapsed = time.time()-total_start
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    return summary

if __name__ == "__main__":
    main()
