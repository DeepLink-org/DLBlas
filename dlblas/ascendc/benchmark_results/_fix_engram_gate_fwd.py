
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from engram_gate_fwd import Model, generate_test_data

so="/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/engram_gate_fwd/operators/engram_gate_fwd/build/libengram_gate_fwd_ops.so"
if not os.path.exists(so):
    print(json.dumps({"op":"engram_gate_fwd","error":"no .so","status":"SKIP"}))
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
        r.append({"label":nm,"shape":[T,H,D],"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4) if au>0 else 0})
    except Exception as e:
        import traceback
        r.append({"label":nm,"shape":[T,H,D],"error":str(e)+" | "+str(traceback.format_exc())[:200],"status":"FAIL"})
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"engram_gate_fwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
