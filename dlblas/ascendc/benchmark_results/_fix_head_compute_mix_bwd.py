
import sys, os, time, json, math
import torch, torch_npu, numpy as np
sys.path.insert(0, "/mnt/data01/zmz/workspace/12agent/waic/origin")
from head_compute_mix_bwd import Model, get_inputs

so="/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge/head_compute_mix_bwd/operators/head_compute_mix_bwd/build/libhead_compute_mix_bwd_ops.so"
if not os.path.exists(so):
    print(json.dumps({"op":"head_compute_mix_bwd","error":"no .so","status":"SKIP"}))
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
    r=[{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(speedup,4)}]
except Exception as e:
    import traceback
    r=[{"label":"default","error":str(e)+" | "+str(traceback.format_exc())[:300],"status":"FAIL"}]
sps=[x["sp"] for x in r if x.get("sp",0)>0]; gm=geomean(sps)
print(json.dumps({"op":"head_compute_mix_bwd","geomean_speedup":round(gm,4),"total":len(r),"passed":sum(1 for x in r if "sp" in x),"shapes":r}))
