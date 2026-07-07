#!/usr/bin/env python3
"""
cannbot-merge 全量算子性能测试 (独立进程,避免.so冲突)
每个算子用最佳可行方法测试: bench脚本 > 独立二进制 > summary.json
"""
import os, sys, time, json, math, subprocess

MERGE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
BENCH_DIR = os.path.join(MERGE, "benchmark_results")
os.chdir(MERGE)
sys.path.insert(0, ORIGIN)

WARMUP, REPEAT = 10, 100

# ===========================================================
# 方法1: 运行 bench 脚本 (已有可用的)
# ===========================================================
WORKING_BENCH_SCRIPTS = {
    "sinkhorn":             "_bench_sinkhorn.py",
    "hc_split_sinkhorn":    "_bench_hc_split_sinkhorn.py",
    "act_quant_kernel":     "_bench_act_quant_kernel.py",
    "expand_kenel_fwd":     "_bench_expand_kenel_fwd.py",
    "apply_mix":            "_bench_apply_mix.py",
    "head_compute_mix_fwd": "_bench_head_compute_mix_fwd.py",
}

WORKING_FIX_SCRIPTS = {
    "engram_fused_weight":  "_fix_engram_fused_weight.py",
    "engram_hash":          "_fix_engram_hash.py",
    "head_compute_mix_bwd": "_fix_head_compute_mix_bwd.py",
}

results = {}

# --- 运行 bench 脚本 ---
for op, script in {**WORKING_BENCH_SCRIPTS, **WORKING_FIX_SCRIPTS}.items():
    sp = os.path.join(BENCH_DIR, script)
    if not os.path.exists(sp): continue
    print(f"[bench] {op}...")
    try:
        out = subprocess.run(["python3", sp], capture_output=True, text=True, timeout=120)
        for line in out.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{') and 'geomean_speedup' in line:
                data = json.loads(line)
                gm = data.get("geomean_speedup", 0)
                shapes = data.get("shapes", [])
                print(f"  => geomean_speedup={gm:.2f}x ({len(shapes)} shapes)")
                results[op] = {"speedup": round(gm, 4), "shapes": shapes, "method": "bench脚本实测"}
                break
        if op not in results:
            print(f"  => no valid output")
    except Exception as e:
        print(f"  => ERROR: {e}")

# ===========================================================
# 方法2: 独立二进制测试 (engram_gate_bwd, expand_kenel_bwd)
# ===========================================================
import numpy as np
import torch, torch_npu

def sync(): torch.npu.synchronize()
def bench(fn, w=WARMUP, r=REPEAT):
    for _ in range(w): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(r): fn()
    sync(); return (time.perf_counter()-t0)/r*1e6

# --- engram_gate_bwd ---
print("\n[binary] engram_gate_bwd...")
try:
    base = f"{MERGE}/engram_gate_bwd/operators/engram_gate_bwd"
    binary = os.path.join(base, "build", "engram_gate_bwd")
    scripts_dir = os.path.join(base, "scripts")
    if os.path.exists(binary) and os.path.exists(scripts_dir):
        os.chdir(os.path.join(base, "build"))
        subprocess.run(["python3", os.path.join(scripts_dir, "gen_data.py")], capture_output=True, timeout=30)
        out = subprocess.run([binary, "14","4","128","1e-6","1e-20"], capture_output=True, text=True, timeout=60)
        au = None
        for line in (out.stdout+out.stderr).split('\n'):
            if 'kernel_time_us' in line:
                try: au = float(line.split('=')[1].strip())
                except: pass
        if au:
            sys.path.insert(0, scripts_dir)
            from golden import compute_golden
            np.random.seed(42)
            go=np.random.randn(14,4,128).astype(np.float32)*0.1
            x=np.random.randn(14,4,128).astype(np.float32)*0.1
            k=np.random.randn(14,4,128).astype(np.float32)*0.1
            v=np.random.randn(14,128).astype(np.float32)*0.1
            wh=np.random.randn(4,128).astype(np.float32)*0.1
            we=np.random.randn(4,128).astype(np.float32)*0.1
            for _ in range(10): compute_golden(go,x,k,v,wh,we)
            t0=time.perf_counter()
            for _ in range(100): compute_golden(go,x,k,v,wh,we)
            tu=(time.perf_counter()-t0)/100*1e6
            sp=tu/au if au>0 else 0
            results["engram_gate_bwd"] = {"speedup": round(sp,4), "torch_us": round(tu,2), "ascendc_us": round(au,2), "method": "独立二进制实测", "shapes":[{"label":"T14H4D128","t_us":round(tu,2),"a_us":round(au,2),"sp":round(sp,4)}]}
            print(f"  => speedup={sp:.2f}x (torch={tu:.1f}us ascendc={au:.1f}us)")
        else:
            print(f"  => no timing")
    else:
        print(f"  => binary or scripts not found")
    os.chdir(MERGE)
except Exception as e:
    print(f"  => ERROR: {e}")
    os.chdir(MERGE)

# ===========================================================
# 方法3: torch.ops 直接测试 (剩余算子)
# ===========================================================
TORCH_OPS_TESTS = {
    "big_fuse": {
        "so_pattern": "libbig_fuse",
        "origin_mod": "big_fuse",
        "op_fn": lambda x: torch.ops.npu.big_fuse(x),
    },
    "engram_gate_fwd": {
        "so_pattern": "libengram_gate_fwd",
        "origin_mod": "engram_gate_fwd",
        "op_fn": None,
    },
    "engram_gate_w_reduce": {
        "so_pattern": "libengram_gate_w_reduce",
        "origin_mod": "engram_gate_w_reduce",
        "op_fn": None,
    },
    "mhc_post": {
        "so_pattern": "libmhc_post",
        "origin_mod": "mhc_post",
        "op_fn": None,
    },
    "norm_fn": {
        "so_pattern": "libnorm_fn",
        "origin_mod": "norm_fn",
        "op_fn": None,
    },
    "pre_split_mixes": {
        "so_pattern": "libpre_split_mixes",
        "origin_mod": "pre_split_mixes",
        "op_fn": None,
    },
    "MTPBlock": {
        "so_pattern": "libmtpblock",
        "origin_mod": "MTPBlock",
        "op_fn": None,
    },
    "sparse_attn": {
        "so_pattern": "libsparse_attn",
        "origin_mod": "sparse_attn",
        "op_fn": None,
    },
    "expand_kenel_bwd": {
        "so_pattern": "libexpand_kenel_bwd",
        "origin_mod": "expand_kenel_bwd",
        "op_fn": None,
    },
    "indexer": {
        "so_pattern": None,  # no .so
        "origin_mod": "indexer",
        "op_fn": None,
    },
}

for op, cfg in TORCH_OPS_TESTS.items():
    if op in results: continue

    print(f"\n[torch.ops] {op}...")

    # find .so
    base = os.path.join(MERGE, op)
    if not os.path.isdir(base):
        print(f"  => dir not found")
        continue

    so_path = None
    for root, dirs, files in os.walk(base):
        for f in files:
            if cfg["so_pattern"] and cfg["so_pattern"] in f and f.endswith(".so"):
                so_path = os.path.join(root, f); break

    if not so_path:
        print(f"  => no .so found")
        continue

    try:
        torch.ops.load_library(so_path)
        mod = __import__(cfg["origin_mod"], fromlist=["Model","get_inputs"])

        # handle different API patterns
        try:
            data = mod.get_inputs()
            m = mod.Model()
        except Exception as e1:
            try:
                data = mod.get_input_groups()
                m = mod.Model()
                data = data[0] if isinstance(data, list) else data
                if isinstance(data, dict):
                    data = list(data.values())
            except Exception as e2:
                print(f"  => cannot get_inputs: {e1}, {e2}")
                continue

        # get torch reference and ascendc op
        try:
            # try torch reference first
            if isinstance(data, (list, tuple)):
                d0 = data[0]
            elif isinstance(data, dict):
                d0 = list(data.values())[0]
            else:
                d0 = data

            if isinstance(d0, torch.Tensor):
                x_npu = d0.npu()
            else:
                x_npu = torch.tensor(d0).npu()

            # try to find the right op name
            op_names = []
            if cfg["op_fn"]:
                op_names.append(("direct", cfg["op_fn"]))
            else:
                # discover op names
                for attr_name in dir(torch.ops.npu):
                    if op.replace("_","")[:6] in attr_name.replace("_","")[:6]:
                        op_names.append((attr_name, lambda x, an=attr_name: getattr(torch.ops.npu, an)(x)))

            if not op_names:
                print(f"  => no op found")
                continue

            # test
            tu = bench(lambda: m.forward(x_npu.cpu()))

            au = None
            for op_name, op_fn in op_names:
                try:
                    au = bench(lambda: op_fn(x_npu))
                    print(f"  => op={op_name} works")
                    break
                except:
                    continue

            if au:
                sp = tu/au if au>0 else 0
                results[op] = {"speedup": round(sp,4), "torch_us": round(tu,2), "ascendc_us": round(au,2), "method": "torch.ops实测", "shapes":[{"label":"default","t_us":round(tu,2),"a_us":round(au,2),"sp":round(sp,4)}]}
                print(f"  => speedup={sp:.2f}x (torch={tu:.1f}us ascendc={au:.1f}us)")
            else:
                print(f"  => no working op found")

        except Exception as e:
            print(f"  => compute ERROR: {type(e).__name__}: {str(e)[:120]}")

    except Exception as e:
        print(f"  => load ERROR: {type(e).__name__}: {str(e)[:120]}")

# ===========================================================
# 汇总输出
# ===========================================================
print("\n" + "=" * 90)
print(" cannbot-merge 全量算子性能测试汇总")
print("=" * 90)
print(f"{'算子':<26} {'加速比':>10} {'AscendC':>12} {'Torch':>12} {'方法':<16}")
print("-" * 80)

all_ops = sorted(set(list(results.keys()) + ["engram_gate_fwd","engram_gate_w_reduce","norm_fn","pre_split_mixes","MTPBlock","sparse_attn","expand_kenel_bwd","indexer"]))

for op in all_ops:
    r = results.get(op, {})
    sp = r.get("speedup", 0)
    au = r.get("ascendc_us", 0)
    tu = r.get("torch_us", 0)
    method = r.get("method", "")

    if not sp:
        # try summary.json
        sf = os.path.join(MERGE, op, "summary.json")
        if not os.path.exists(sf):
            for root, dirs, files in os.walk(os.path.join(MERGE, op)):
                if "summary.json" in files:
                    sf = os.path.join(root, "summary.json"); break
        if os.path.exists(sf):
            with open(sf) as f: d = json.load(f)
            perf = d.get("perf_data", {})
            sp = perf.get("speedup_vs_torch") or perf.get("geomean_speedup_vs_torch") or perf.get("speedup_vs_torch_cpu") or 0
            au = perf.get("ascend_us") or perf.get("ascendc_kernel_us") or perf.get("ascendc_us") or 0
            tu = perf.get("torch_us") or perf.get("torch_cpu_ref_us") or perf.get("torch_npu_us") or 0
            if isinstance(au, str): au = 0
            if isinstance(tu, str): tu = 0
            method = "summary.json"

    sp_s = f"{float(sp):.2f}x" if sp else "-"
    au_s = f"{float(au):.1f}" if au else "-"
    tu_s = f"{float(tu):.1f}" if tu else "-"
    flag = "🟢" if (isinstance(sp, (int,float)) and sp > 1.0) else "🔴"
    print(f"{flag} {op:<24} {sp_s:>10} {au_s:>12} {tu_s:>12} {method:<16}")

# 统计
measured = sum(1 for r in results.values() if r.get("speedup",0) > 0)
total = len(all_ops)
fast = sum(1 for r in results.values() if r.get("speedup",0) > 1.0)
slow = sum(1 for r in results.values() if 0 < r.get("speedup",0) <= 1.0)

print(f"\n实测: {measured}/{total}  |  🟢加速: {fast}  |  🔴减速: {slow}")

# Save
out_path = os.path.join(BENCH_DIR, "full_retest_results.json")
with open(out_path, "w") as f:
    json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2, ensure_ascii=False)
print(f"结果已保存: {out_path}")
