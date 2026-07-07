#!/usr/bin/env python3
"""Fix and benchmark remaining operators individually."""
import subprocess, sys, os, json, time, math, shutil

BASE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
OUT = os.path.join(BASE, "benchmark_results")

def run_bench(op_name, code, timeout=180):
    """Write and run a benchmark script, return parsed result."""
    path = os.path.join(OUT, f"_fix_{op_name}.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"  [{op_name}] Running..."); sys.stdout.flush()
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-u", path], capture_output=True, text=True, timeout=timeout,
                          cwd=BASE, env={**os.environ, "PYTHONPATH": f"{ORIGIN}:{os.environ.get('PYTHONPATH','')}"})
        elapsed = time.time()-t0
        stdout = r.stdout.strip()
        stderr = r.stderr.strip()
        if stderr and any(k in stderr.lower() for k in ['error','traceback','fail','terminate']):
            last_lines = stderr.split('\n')[-10:]
            print(f"    stderr: {' | '.join(last_lines)[:300]}")
        if stdout:
            for line in stdout.split('\n'):
                line = line.strip()
                if line.startswith('{'):
                    try:
                        res = json.loads(line)
                        res["elapsed_s"] = round(elapsed, 1)
                        res["returncode"] = r.returncode
                        return res
                    except: pass
        return {"op": op_name, "error": f"no valid JSON. rc={r.returncode}", "stdout": stdout[:300], "stderr": stderr[:300], "status": "ERROR", "elapsed_s": round(elapsed,1)}
    except subprocess.TimeoutExpired:
        return {"op": op_name, "error": "timeout", "status": "TIMEOUT", "elapsed_s": time.time()-t0}
    except Exception as e:
        return {"op": op_name, "error": str(e), "status": "ERROR", "elapsed_s": time.time()-t0}

# ===================================================================
# Fixed operator scripts
# ===================================================================

results = {}

# --- engram_fused_weight (fix: Model needs hc_mult, hidden_size) ---
results["engram_fused_weight"] = run_bench("engram_fused_weight", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from engram_fused_weight import Model, generate_test_data

torch.ops.load_library("{BASE}/engram_fused_weight/build/libengram_fused_weight_ops.so")

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
for nm,H,D in [("small",16,128),("default",64,1280),("medium",128,1280)]:
    try:
        wh,we=generate_test_data(H,D); m=Model(H,D)
        wh_n,we_n=wh.npu(),we.npu()
        tu=bench(lambda:m.forward(wh,we)); au=bench(lambda:torch.ops.npu.engram_fused_weight(wh_n,we_n))
        r.append({{"label":nm,"shape":[H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0}})
    except Exception as e:
        r.append({{"label":nm,"shape":[H,D],"error":str(e),"status":"FAIL"}})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"engram_fused_weight","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- engram_hash (fix: dict in generate_test_data) ---
results["engram_hash"] = run_bench("engram_hash", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from engram_hash import Model, generate_test_data

torch.ops.load_library("{BASE}/engram_hash/build/libengram_hash_ops.so")

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
        r.append({{"label":nm,"shape":[nt,N,L,T],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0}})
    except Exception as e:
        r.append({{"label":nm,"shape":[nt,N,L,T],"error":str(e),"status":"FAIL"}})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"engram_hash","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- engram_gate_fwd (fix: params key name, forward args) ---
results["engram_gate_fwd"] = run_bench("engram_gate_fwd", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from engram_gate_fwd import Model, generate_test_data

so="{BASE}/engram_gate_fwd/operators/engram_gate_fwd/build/libengram_gate_fwd_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"engram_gate_fwd","error":"no .so","status":"SKIP"}}))
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

r=[]
for nm,T,H,D in [("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128)]:
    try:
        params = dict(num_tokens=T, hc=H, hidden_size=D)
        hs,k,v,wh,we=generate_test_data(params); m=Model()
        hs_n,k_n,v_n,wh_n,we_n=hs.npu(),k.npu(),v.npu(),wh.npu(),we.npu()
        # Model forward: hidden_states, k, v, weight_hidden, weight_embed, clamp_value, eps
        tu=bench(lambda:m.forward(hs,k,v,wh,we,1.0,0.01))
        au=bench(lambda:torch.ops.npu.engram_gate_fwd(hs_n,k_n,v_n,wh_n,we_n,1.0,0.01))
        r.append({{"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0}})
    except Exception as e:
        import traceback
        r.append({{"label":nm,"shape":[T,H,D],"error":str(e)+" | "+str(traceback.format_exc())[:200],"status":"FAIL"}})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"engram_gate_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- engram_gate_bwd (fix: check forward signature) ---
results["engram_gate_bwd"] = run_bench("engram_gate_bwd", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from engram_gate_bwd import Model, generate_test_data

torch.ops.load_library("{BASE}/engram_gate_bwd/build/libengram_gate_bwd_ops.so")

WARMUP=5; REPEAT=50
def sync(): torch.npu.synchronize()
def bench(fn,w=WARMUP,r=REPEAT):
    for _ in range(w): fn()
    sync(); t0=time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6

def geomean(v):
    vv=[x for x in v if x>0]
    return math.exp(sum(math.log(x) for x in vv)/len(vv)) if vv else 0

# Check signatures first
import inspect
sig_fwd = inspect.signature(Model.forward)
n_params = len(sig_fwd.parameters) - 1  # minus self
print(f"Model.forward takes {{n_params}} args: {{list(sig_fwd.parameters.keys())}}", file=sys.stderr)

r=[]
for nm,T,H,D in [("T1H4D128",1,4,128),("T8H4D128",8,4,128),("T14H4D128",14,4,128)]:
    try:
        data=generate_test_data(T,H,D)
        print(f"generate_test_data returned {{len(data)}} items", file=sys.stderr)
        m=Model()
        # Move data to NPU
        data_npu = [d.npu() if isinstance(d,torch.Tensor) else d for d in data]
        tu=bench(lambda:m.forward(*data))
        au=bench(lambda:torch.ops.npu.engram_gate_bwd(*data_npu))
        r.append({{"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0}})
    except Exception as e:
        import traceback
        r.append({{"label":nm,"shape":[T,H,D],"error":str(e),"status":"FAIL"}})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"engram_gate_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- engram_gate_w_reduce (fix: forward takes 5 args) ---
results["engram_gate_w_reduce"] = run_bench("engram_gate_w_reduce", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from engram_gate_w_reduce import Model, generate_test_data

so="{BASE}/engram_gate_w_reduce/operators/engram_gate_w_reduce/build/libengram_gate_w_reduce_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"engram_gate_w_reduce","error":"no .so","status":"SKIP"}}))
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

# Model.forward(self, grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
r=[]
for nm,D in [("default",128),("small",64)]:
    try:
        data=generate_test_data(D); m=Model()
        data_npu = [d.npu() if isinstance(d,torch.Tensor) else d for d in data]
        tu=bench(lambda:m.forward(*data)); au=bench(lambda:torch.ops.npu.engram_gate_w_reduce(*data_npu))
        r.append({{"label":nm,"shape":[D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0}})
    except Exception as e:
        import traceback
        r.append({{"label":nm,"shape":[D],"error":str(e),"status":"FAIL"}})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"engram_gate_w_reduce","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- head_compute_mix_bwd (fix: forward takes 4 args) ---
results["head_compute_mix_bwd"] = run_bench("head_compute_mix_bwd", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from head_compute_mix_bwd import Model, get_inputs

so="{BASE}/head_compute_mix_bwd/operators/head_compute_mix_bwd/build/libhead_compute_mix_bwd_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"head_compute_mix_bwd","error":"no .so","status":"SKIP"}}))
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

# Model.forward(self, input_mix, mhc_scale, mhc_base, grad_out)
r=[]
try:
    data=get_inputs(); m=Model()
    # get_inputs returns: input_mix, mhc_scale, mhc_base, mhc_pre_eps
    # But forward needs: input_mix, mhc_scale, mhc_base, grad_out
    # So we need grad_out from somewhere - use data[0] with grad
    im,ms,mb,eps = data[:4]
    # Create grad_out same shape as output
    with torch.no_grad():
        out = m.forward(im, ms, mb, im)  # use im as grad_out placeholder
    go = torch.randn_like(out) if isinstance(out, torch.Tensor) else torch.randn_like(im)
    im_n,ms_n,mb_n,go_n = im.npu(),ms.npu(),mb.npu(),go.npu()
    tu=bench(lambda:m.forward(im,ms,mb,go)); au=bench(lambda:torch.ops.npu.head_compute_mix_bwd(im_n,ms_n,mb_n,go_n))
    speedup=tu/au if au>0 else 0
    r=[{{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}}]
except Exception as e:
    import traceback
    r=[{{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"head_compute_mix_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- expand_kenel_bwd ---
results["expand_kenel_bwd"] = run_bench("expand_kenel_bwd", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from expand_kenel_bwd import Model, get_inputs

so="{BASE}/expand_kenel_bwd/operators/expand_kenel_bwd/build/libexpand_kenel_bwd_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"expand_kenel_bwd","error":"no .so","status":"SKIP"}}))
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

r=[]
try:
    data=get_inputs(); m=Model()
    o_grad=data[0]; o_n=o_grad.npu()
    tu=bench(lambda:m.forward(o_grad)); au=bench(lambda:torch.ops.npu.expand_kenel_bwd(o_n))
    speedup=tu/au if au>0 else 0
    r=[{{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}}]
except Exception as e:
    import traceback
    r=[{{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"expand_kenel_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- pre_split_mixes (fix: Model needs mhc_mult) ---
results["pre_split_mixes"] = run_bench("pre_split_mixes", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from pre_split_mixes import Model, get_inputs

torch.ops.load_library("{BASE}/pre_split_mixes/build/libpre_split_mixes_ops.so")

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
try:
    data=get_inputs(); m=Model(4)  # mhc_mult=4
    im=data[0]; im_n=im.npu()
    tu=bench(lambda:m.forward(im)); au=bench(lambda:torch.ops.npu.pre_split_mixes(im_n,*[d.npu() if isinstance(d,torch.Tensor) else d for d in data[1:]]))
    speedup=tu/au if au>0 else 0
    r=[{{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}}]
except Exception as e:
    import traceback
    r=[{{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"pre_split_mixes","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- big_fuse (fix: Model needs mhc_mult, hidden_size) ---
results["big_fuse"] = run_bench("big_fuse", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from big_fuse import Model, get_inputs

so="{BASE}/big_fuse/operators/big_fuse/build/libbig_fuse_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"big_fuse","error":"no .so","status":"SKIP"}}))
    sys.exit(0)
torch.ops.load_library(so)

WARMUP=5; REPEAT=50
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
try:
    data=get_inputs(); m=Model(4, 1280)  # mhc_mult=4, hidden_size=1280
    residual=data[0]; residual_n=residual.npu()
    # Model.forward takes (self, residual) - returns tuple
    # torch.ops.npu.big_fuse expects (residual, fn, mhc_scale, mhc_base) -> Tensor[]
    # get_inputs returns: residual, fn, mhc_scale, mhc_base
    tu=bench(lambda:m.forward(residual))
    fn_n,ms_n,mb_n = data[1].npu(),data[2].npu(),data[3].npu()
    au=bench(lambda:torch.ops.npu.big_fuse(residual_n,fn_n,ms_n,mb_n))
    speedup=tu/au if au>0 else 0
    r=[{{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}}]
except Exception as e:
    import traceback
    r=[{{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"big_fuse","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- mhc_post ---
results["mhc_post"] = run_bench("mhc_post", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")
from mhc_post import Model, generate_mhc_post_test_data

so="{BASE}/mhc_post/operators/mhc_post/build/libmhc_post_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"mhc_post","error":"no .so","status":"SKIP"}}))
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

r=[]
for nm,B,S,M,H in [("default",1,1024,4,1280),("small",1,64,2,640)]:
    try:
        x,res,plm,comb=generate_mhc_post_test_data(B,S,M,H); m=Model()
        x_n,res_n,plm_n,comb_n=x.npu(),res.npu(),plm.npu(),comb.npu()
        tu=bench(lambda:m.forward(x,res,plm,comb)); au=bench(lambda:torch.ops.npu.mhc_post(x_n,res_n,plm_n,comb_n))
        r.append({{"label":nm,"shape":[B,S,M,H],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0}})
    except Exception as e:
        import traceback
        r.append({{"label":nm,"shape":[B,S,M,H],"error":str(e),"status":"FAIL"}})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"mhc_post","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- MTPBlock (simple hc_post test) ---
results["MTPBlock"] = run_bench("MTPBlock", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")

so="{BASE}/MTPBlock/build/libmtpblock_ops.so"
if not os.path.exists(so):
    print(json.dumps({{"op":"MTPBlock","error":"no .so","status":"SKIP"}}))
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
print(f"mtpblock ops: {{ops}}", file=sys.stderr)

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
    r=[{{"label":"default","shape":[B,S,M,H],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}}]
except Exception as e:
    import traceback
    r=[{{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({{"op":"MTPBlock","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}}))
''')

# --- sparse_attn (try with Scalar instead of float, or try softmax_scale as tensor) ---
results["sparse_attn"] = run_bench("sparse_attn", f'''
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "{ORIGIN}")

# The .so has schema conflict. Try to work around it.
so="{BASE}/sparse_attn/build/libsparse_attn_ops.so"

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
alt_so = "{BASE}/sparse_attn/operators/sparse_attn/build/libsparse_attn_ops.so"
if os.path.exists(alt_so):
    so = alt_so
    print(f"Using alt .so: {{so}}", file=sys.stderr)

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
            r=[{{"label":"default","error":f".so schema conflict: {{e2}}","status":"FAIL"}}]
            sps=[]; gm=0
            print(json.dumps({{"op":"sparse_attn","geomean_speedup":0,"total":1,"passed":0,"shapes":r,"error":str(e2)}}))
            sys.exit(0)

    speedup=tu/au if au>0 else 0
    r=[{{"label":"default","shape":[b,m,n,h,d,tk],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}}]
    sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
    print(json.dumps({{"op":"sparse_attn","geomean_speedup":round(gm,4),"total":len(r),"passed":len(r),"shapes":r}}))
except Exception as e:
    import traceback
    print(json.dumps({{"op":"sparse_attn","error":str(e),"trace":traceback.format_exc()[:500],"status":"ERROR"}}))
''')

# --- indexer (no AscendC .so, torch-only reference) ---
results["indexer"] = {"op": "indexer", "geomean_speedup": None, "total": 0, "passed": 0,
                      "shapes": [], "note": "No AscendC .so. indexer uses torch.cuda - incompatible with NPU."}

# --- norm_fn (already got 23.61x with 1/3 passed, keep existing) ---
# Already captured in the first run

print("\n\n=== Fixed Operator Results ===")
all_sp = []
for nm, res in sorted(results.items()):
    gm = res.get("geomean_speedup")
    status = "OK" if gm and gm > 0 else res.get("status", "UNKNOWN")
    print(f"  [{nm}] geomean={gm} status={status}")
    if gm and gm > 0:
        all_sp.append((nm, gm))
    # Write result
    with open(f"{OUT}/{nm}_result.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

if all_sp:
    all_sp.sort(key=lambda x: x[1], reverse=True)
    for nm, sp in all_sp:
        print(f"    {nm}: {sp:.4f}x")

print(f"\nFixed {len(results)} operators, {len(all_sp)} with valid speedup")
