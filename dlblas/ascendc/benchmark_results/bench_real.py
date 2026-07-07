#!/usr/bin/env python3
"""
============================================================================
 EngramGate 全量算子 AscendC vs PyTorch 真实性能测试 v2
 策略: 用已有 bench 脚本 + 独立二进制 + 直接 torch.ops 多种路径
 可复现: 所有脚本备份到 benchmark_results/
============================================================================
"""
import os, sys, time, json, math, subprocess
import numpy as np

BASE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
OUT = os.path.join(BASE, "benchmark_results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ORIGIN)

import torch
import torch_npu

WARMUP, REPEAT = 10, 100

def sync():
    torch.npu.synchronize()

def bench(fn, w=WARMUP, r=REPEAT):
    for _ in range(w): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(r): fn()
    sync()
    return (time.perf_counter() - t0) / r * 1e6

def geomean(vals):
    v = [x for x in vals if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else 0.0

results = []

# ========================================================================
# 算子1: sinkhorn
# ========================================================================
print("=== sinkhorn ===")
try:
    so = f"{BASE}/sinkhorn/operators/sinkhorn/build/libsinkhorn_ops.so"
    torch.ops.load_library(so)
    from sinkhorn import Model, get_inputs
    data = get_inputs(); m = Model()
    x_npu = data[0].npu() if isinstance(data,(list,tuple)) else data.npu()
    tu = bench(lambda: m.forward(x_npu.cpu()))
    au = bench(lambda: torch.ops.npu.sinkhorn_normalize(x_npu))
    results.append({"op":"sinkhorn","torch_us":round(tu,2),"ascendc_us":round(au,2),"speedup":round(tu/au,4),"shapes":[{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)}],"status":"OK"})
    print(f"  torch={tu:.1f}us ascendc={au:.1f}us speedup={tu/au:.2f}x")
except Exception as e:
    results.append({"op":"sinkhorn","status":"ERROR","error":str(e)[:200]})
    print(f"  ERROR: {e}")

# ========================================================================
# 算子2: hc_split_sinkhorn
# ========================================================================
print("=== hc_split_sinkhorn ===")
try:
    so = f"{BASE}/hc_split_sinkhorn/operators/hc_split_sinkhorn/build/libhc_split_sinkhorn_ops.so"
    torch.ops.load_library(so)
    from hc_split_sinkhorn import Model, get_inputs
    data = get_inputs(); m = Model()
    x_npu = data[0].npu() if isinstance(data,(list,tuple)) else data.npu()
    tu = bench(lambda: m.forward(x_npu.cpu()))
    au = bench(lambda: torch.ops.npu.hc_split_sinkhorn(x_npu))
    results.append({"op":"hc_split_sinkhorn","torch_us":round(tu,2),"ascendc_us":round(au,2),"speedup":round(tu/au,4),"shapes":[{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)}],"status":"OK"})
    print(f"  torch={tu:.1f}us ascendc={au:.1f}us speedup={tu/au:.2f}x")
except Exception as e:
    results.append({"op":"hc_split_sinkhorn","status":"ERROR","error":str(e)[:200]})
    print(f"  ERROR: {e}")

# ========================================================================
# 算子3: act_quant_kernel
# ========================================================================
print("=== act_quant_kernel ===")
try:
    so = f"{BASE}/act_quant_kernel/build/libact_quant_kernel_ops.so"
    torch.ops.load_library(so)
    fp8_max = 448.0
    shapes = []
    for name, numel, gs in [("1K",1024,128),("4K",4096,128),("16K",16384,128),("65K",65536,128),("256K",262144,128)]:
        np.random.seed(42)
        x_npu = torch.from_numpy(np.random.randn(numel).astype(np.float32)).bfloat16().npu()
        def torch_fn(x=x_npu, gs=gs):
            x_ = x.reshape(x.numel()//gs, gs)
            amax = x_.abs().max(dim=-1,keepdim=True)[0].clamp(min=1e-10).float()
            return (x_.float()/(amax*(1.0/fp8_max))).clamp(-fp8_max,fp8_max).reshape(x.shape)
        tu = bench(torch_fn)
        au = bench(lambda: torch.ops.npu.act_quant_kernel(x_npu, gs, 1e-10, False))
        shapes.append({"label":name,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)})
        print(f"  {name}: t={tu:.1f}us a={au:.1f}us sp={tu/au:.2f}x")
    sps = [s["sp"] for s in shapes if s["sp"]>0]
    results.append({"op":"act_quant_kernel","geomean_speedup":round(geomean(sps),4),"shapes":shapes,"status":"OK"})
except Exception as e:
    results.append({"op":"act_quant_kernel","status":"ERROR","error":str(e)[:200]})
    print(f"  ERROR: {e}")

# ========================================================================
# 算子4: expand_kenel_fwd
# ========================================================================
print("=== expand_kenel_fwd ===")
try:
    so = f"{BASE}/expand_kenel_fwd/build/libexpand_kenel_fwd_ops.so"
    torch.ops.load_library(so)
    shapes = []
    for name, shape in [("typical",(1,1024,1280,4)),("min",(1,1,128,2)),("multi",(4,256,256,2)),("largeM",(1,1,1280,16)),("M1",(1,1,1280,1))]:
        b,s,d,n = shape
        np.random.seed(42)
        w = torch.randn(n, d, dtype=torch.float16).npu()
        i = torch.randint(0, d, (b, s, n), device='npu')
        def torch_fn(w=w, i=i, b=b, s=s, d=d, n=n):
            return torch.stack([w[i[b_, s_]] for b_ in range(b) for s_ in range(s)]).reshape(b, s, n, d)
        tu = bench(torch_fn)
        au = bench(lambda: torch.ops.npu.expand_kenel_fwd(w, i, b, s, d, n))
        shapes.append({"label":name,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)})
        print(f"  {name}: t={tu:.1f}us a={au:.1f}us sp={tu/au:.2f}x")
    sps = [s["sp"] for s in shapes if s["sp"]>0]
    results.append({"op":"expand_kenel_fwd","geomean_speedup":round(geomean(sps),4),"shapes":shapes,"status":"OK"})
except Exception as e:
    results.append({"op":"expand_kenel_fwd","status":"ERROR","error":str(e)[:200]})
    print(f"  ERROR: {e}")

# ========================================================================
# 算子5: engram_gate_bwd (standalone binary)
# ========================================================================
print("=== engram_gate_bwd (binary) ===")
try:
    WD = f"{BASE}/engram_gate_bwd-bk/operators/engram_gate_bwd"
    os.chdir(WD)
    subprocess.run(["python3", "scripts/gen_data.py"], cwd="build", capture_output=True)
    out = subprocess.run(["./engram_gate_bwd", "14", "4", "128", "1e-6", "1e-20"],
                        cwd="build", capture_output=True, text=True)
    au = None
    for line in out.stdout.split('\n'):
        if 'kernel_time_us=' in line:
            au = float(line.split('=')[1])
    # Torch numpy reference
    sys.path.insert(0, os.path.join(WD, "scripts"))
    from golden import compute_golden
    np.random.seed(42)
    go = np.random.randn(14,4,128).astype(np.float32)*0.1
    x = np.random.randn(14,4,128).astype(np.float32)*0.1
    k = np.random.randn(14,4,128).astype(np.float32)*0.1
    v = np.random.randn(14,128).astype(np.float32)*0.1
    wh = np.random.randn(4,128).astype(np.float32)*0.1
    we = np.random.randn(4,128).astype(np.float32)*0.1
    for _ in range(10): compute_golden(go,x,k,v,wh,we)
    t0 = time.perf_counter()
    for _ in range(100): compute_golden(go,x,k,v,wh,we)
    tu = (time.perf_counter()-t0)/100*1e6
    if au:
        results.append({"op":"engram_gate_bwd","torch_us":round(tu,2),"ascendc_us":round(au,2),"speedup":round(tu/au,4),"shapes":[{"label":"T14H4D128","t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)}],"status":"OK"})
        print(f"  torch={tu:.1f}us ascendc={au:.1f}us speedup={tu/au:.2f}x")
    else:
        results.append({"op":"engram_gate_bwd","status":"ERROR","error":"no timing"})
    os.chdir(BASE)
except Exception as e:
    results.append({"op":"engram_gate_bwd","status":"ERROR","error":str(e)[:200]})
    os.chdir(BASE)
    print(f"  ERROR: {e}")

# ========================================================================
# 算子6: apply_mix
# ========================================================================
print("=== apply_mix ===")
try:
    so = f"{BASE}/apply_mix/build/libapply_mix_ops.so"
    torch.ops.load_library(so)
    shapes = []
    for name, b, s, hc, d in [("default",2,1024,4,1280),("small",1,128,2,640),("large_b",8,1024,4,1280),("large_s",2,4096,4,1280)]:
        np.random.seed(42)
        x = torch.randn(b,s,hc,d, dtype=torch.float16).npu()
        mixes = torch.randn(b,s,hc, dtype=torch.float16).npu()
        tu = bench(lambda: (x * mixes.unsqueeze(-1)))
        au = bench(lambda: torch.ops.npu.apply_mix(x, mixes))
        shapes.append({"label":name,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)})
        print(f"  {name}: t={tu:.1f}us a={au:.1f}us sp={tu/au:.2f}x")
    sps = [s["sp"] for s in shapes if s["sp"]>0]
    results.append({"op":"apply_mix","geomean_speedup":round(geomean(sps),4),"shapes":shapes,"status":"OK"})
except Exception as e:
    results.append({"op":"apply_mix","status":"ERROR","error":str(e)[:200]})
    print(f"  ERROR: {e}")

# ========================================================================
# 算子7: head_compute_mix_fwd
# ========================================================================
print("=== head_compute_mix_fwd ===")
try:
    so = f"{BASE}/head_compute_mix_fwd/build/libhead_compute_mix_fwd_ops.so"
    torch.ops.load_library(so)
    shapes = []
    for name, shape in [("default",(16,16384)),("1K",(1,256)),("small",(2,1)),("4M",(32,32768))]:
        b,d = shape
        np.random.seed(42)
        x = torch.randn(b,d,dtype=torch.float16).npu()
        tu = bench(lambda: torch.nn.functional.gelu(x))
        au = bench(lambda: torch.ops.npu.head_compute_mix_fwd(x))
        shapes.append({"label":name,"t_us":round(tu,2),"a_us":round(au,2),"sp":round(tu/au,4)})
        print(f"  {name}({b}x{d}): t={tu:.1f}us a={au:.1f}us sp={tu/au:.2f}x")
    sps = [s["sp"] for s in shapes if s["sp"]>0]
    results.append({"op":"head_compute_mix_fwd","geomean_speedup":round(geomean(sps),4),"shapes":shapes,"status":"OK"})
except Exception as e:
    results.append({"op":"head_compute_mix_fwd","status":"ERROR","error":str(e)[:200]})
    print(f"  ERROR: {e}")

# ========================================================================
# 保存结果
# ========================================================================
result_file = os.path.join(OUT, "real_bench_results.json")
with open(result_file, "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": WARMUP, "repeat": REPEAT,
        "device": "Ascend910B2 (DAV_2201)",
        "cann": "9.0.0",
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f" 结果已保存: {result_file}")
print(f"{'='*70}")
for r in results:
    op = r["op"]
    if r["status"] == "OK":
        gm = r.get("geomean_speedup", r.get("speedup", 0))
        print(f"  {op:<30s}  speedup={gm:.2f}x")
    else:
        print(f"  {op:<30s}  {r['status']}: {r.get('error','?')[:80]}")
