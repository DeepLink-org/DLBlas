#!/usr/bin/env python3
"""
纯 Kernel 耗时测试脚本 (排除 launch overhead)

方法:
  方法1 (首选): 修改 host 代码注入计时 → 重编译 → 运行二进制 → 提取 kernel_time_us
  方法2 (备选): torch.npu.profiler → 导出 chrome trace → 解析 kernel 事件
  方法3 (兜底): summary.json 中的 msprof 历史数据

输出: kernel_time_result.json + 对比报告
用法: python3 bench_kernel_only_v2.py
"""

import os, sys, re, json, time, subprocess, shutil, tempfile
from collections import OrderedDict

MERGE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
sys.path.insert(0, ORIGIN)

WARMUP, REPEAT = 10, 100

# ============================================================================
# 算子定义
# ============================================================================
# 每个算子的配置
OP_CONFIGS = OrderedDict([
    # ---- Method 1 算子 (有 run.sh + host 文件，可注入计时) ----
    ("engram_gate_bwd", {
        "method": 1,
        "binary_path": None,  # auto-detect
        "build_cmd": "run_sh",  # use run.sh
        "run_args": ["14", "4", "128", "1e-6", "1e-20"],
        "gen_data_cmd": ["python3", "scripts/gen_data.py"],
    }),
    ("expand_kenel_fwd", {
        "method": 1,
        "run_args": ["1", "1024", "1280", "4", "fp16"],
    }),
    ("apply_mix", {
        "method": 1,
        "run_args": [],
    }),
    ("big_fuse", {
        "method": 1,
        "run_args": [],
    }),
    ("engram_fused_weight", {
        "method": 1,
        "run_args": [],
    }),
    ("engram_gate_fwd", {
        "method": 1,
        "run_args": ["14", "4", "128", "fp16"],
    }),
    ("engram_gate_w_reduce", {
        "method": 1,
        "run_args": ["14", "4", "128", "fp16"],
    }),
    ("expand_kenel_bwd", {
        "method": 1,
        "run_args": ["1", "1024", "1280", "4", "fp16"],
    }),
    ("hc_split_sinkhorn", {
        "method": 1,
        "run_args": [],
    }),
    ("head_compute_mix_bwd", {
        "method": 1,
        "run_args": [],
    }),
    ("head_compute_mix_fwd", {
        "method": 1,
        "run_args": [],
    }),
    ("mhc_post", {
        "method": 1,
        "run_args": [],
    }),
    ("norm_fn", {
        "method": 1,
        "run_args": [],
    }),
    ("pre_split_mixes", {
        "method": 1,
        "run_args": [],
    }),
    ("sparse_attn", {
        "method": 1,
        "run_args": [],
    }),
    ("MTPBlock", {
        "method": 1,
        "run_args": [],
    }),
    ("sinkhorn", {
        "method": 1,
        "run_args": [],
    }),
    # ---- Method 2 算子 (torch.ops profiler) ----
    ("act_quant_kernel", {
        "method": 2,
        "so_pattern": "libact_quant_kernel",
    }),
    # ---- Method 3 算子 (summary.json) ----
    ("engram_hash", {
        "method": 3,
    }),
    ("indexer", {
        "method": 3,
    }),
])


# ============================================================================
# 工具函数
# ============================================================================
def find_op_dir(op):
    """找到算子的实际工作目录"""
    d = os.path.join(MERGE, op)
    if os.path.isdir(d):
        # 检查 operators/<op> 子目录
        sub = os.path.join(d, "operators", op)
        if os.path.isdir(sub):
            return sub
        return d
    return None


def find_host_file(op_dir):
    """找到包含 kernel launch 的 host 文件"""
    for root, dirs, files in os.walk(op_dir):
        # 跳过 build 目录
        dirs[:] = [d for d in dirs if d not in ("build", "build_review", "__pycache__")]
        for f in files:
            if not f.endswith(".asc"):
                continue
            if "kernel" in f or "diag" in f or "data_utils" in f:
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath) as fh:
                    content = fh.read(4096)  # read first 4K
                if "_kernel<<<" in content or "KernelCall" in content or "KernelLaunch" in content:
                    return fpath
            except:
                pass
    return None


def find_run_sh(op_dir):
    """找到 run.sh"""
    for p in [os.path.join(op_dir, "run.sh"),
              os.path.join(os.path.dirname(op_dir), "run.sh")]:
        if os.path.isfile(p):
            return p
    return None


def find_binary(op_dir, op_name):
    """找到已编译的二进制文件"""
    for root, dirs, files in os.walk(op_dir):
        if op_name in files:
            fpath = os.path.join(root, op_name)
            if os.access(fpath, os.X_OK):
                return fpath
    return None


def inject_timing(host_path):
    """向 host 文件注入 kernel 计时代码，返回是否修改"""
    with open(host_path) as f:
        content = f.read()

    # 检查是否已有计时
    if "kernel_time_us" in content:
        return False

    # 找 kernel launch 行
    lines = content.split('\n')
    new_lines = []
    kernel_line_idx = -1
    sync_after_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_kernel = (('_kernel<<<' in stripped or 'KernelCall(' in stripped or
                      'KernelLaunch(' in stripped) and not stripped.startswith('//'))

        if is_kernel and kernel_line_idx < 0:
            kernel_line_idx = i

    if kernel_line_idx < 0:
        return False

    # 在 kernel launch 之后找 aclrtSynchronizeStream
    for i in range(kernel_line_idx + 1, min(kernel_line_idx + 30, len(lines))):
        if 'aclrtSynchronizeStream' in lines[i]:
            sync_after_idx = i
            break

    if sync_after_idx < 0:
        return False

    # 注入: sync before → timer start → kernel → sync → timer end → printf
    indent = lines[kernel_line_idx][:len(lines[kernel_line_idx]) - len(lines[kernel_line_idx].lstrip())]
    sync_indent = lines[sync_after_idx][:len(lines[sync_after_idx]) - len(lines[sync_after_idx].lstrip())]

    # kernel launch 前插入
    kernel_line = lines[kernel_line_idx]
    lines[kernel_line_idx] = (f'{indent}aclrtSynchronizeStream(stream);\n'
                              f'{indent}auto _t0 = std::chrono::high_resolution_clock::now();\n'
                              f'{kernel_line}')

    # 调整后续索引（插入了2行）
    sync_after_idx += 2

    # sync 后插入
    sync_line = lines[sync_after_idx]
    lines[sync_after_idx] = (f'{sync_line}\n'
                             f'{sync_indent}auto _t1 = std::chrono::high_resolution_clock::now();\n'
                             f'{sync_indent}double _kernel_us = std::chrono::duration<double, std::micro>(_t1 - _t0).count();\n'
                             f'{sync_indent}printf("[PERF] kernel_time_us=%.2f\\n", _kernel_us);')

    # 添加 #include <chrono>
    if "#include <chrono>" not in content and "<chrono>" not in content:
        last_include = content.rfind("#include")
        end_of_line = content.find("\n", last_include)
        new_content = '\n'.join(lines)
        new_content = (new_content[:end_of_line+1] +
                       '#include <chrono>\n' +
                       new_content[end_of_line+1:])
    else:
        new_content = '\n'.join(lines)

    with open(host_path, 'w') as f:
        f.write(new_content)
    return True


def build_with_run_sh(op_dir, op_name, skip_build=False):
    """使用 run.sh 编译（处理 cmake 路径问题），返回 build_dir"""
    run_sh = find_run_sh(op_dir)
    if not run_sh:
        return None

    build_dir = os.path.join(op_dir, "build")
    if not os.path.isdir(build_dir):
        build_dir = os.path.join(os.path.dirname(op_dir), "build")
    if not os.path.isdir(build_dir):
        # 尝试创建 build 并运行 cmake
        os.makedirs(build_dir, exist_ok=True)

    run_dir = os.path.dirname(run_sh)

    if not skip_build:
        # 删除 CMakeCache.txt 以清除 stale 路径
        for cache in [os.path.join(build_dir, "CMakeCache.txt"),
                      os.path.join(run_dir, "build", "CMakeCache.txt")]:
            if os.path.isfile(cache):
                os.remove(cache)

        # 运行 cmake
        os.chdir(build_dir)
        src_dir = run_dir
        r = subprocess.run(["cmake", src_dir, "-DCMAKE_BUILD_TYPE=Release"],
                          capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            # 尝试从 operators 目录 cmake
            alt_src = os.path.join(run_dir, "..")
            r = subprocess.run(["cmake", alt_src, "-DCMAKE_BUILD_TYPE=Release"],
                              capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                return None, f"cmake failed: {r.stderr[-300:]}"

        r = subprocess.run(["make", "-j4"], capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return None, f"make failed: {r.stderr[-300:]}"

    return build_dir, None


def run_binary_and_extract(op_dir, op_name, build_dir, run_args):
    """运行二进制并提取 kernel_time_us"""
    binary = find_binary(op_dir, op_name)
    if not binary and build_dir:
        binary = os.path.join(build_dir, op_name)

    if not binary or not os.path.isfile(binary):
        return None, f"binary not found"

    # 先运行 gen_data.py (如果存在)
    gen_data_paths = [
        os.path.join(os.path.dirname(build_dir), "scripts", "gen_data.py"),
        os.path.join(op_dir, "scripts", "gen_data.py"),
    ]
    for gdp in gen_data_paths:
        if os.path.isfile(gdp):
            try:
                subprocess.run(["python3", gdp] + list(run_args),
                              cwd=build_dir, capture_output=True, timeout=30)
            except:
                pass
            break

    # 运行二进制
    try:
        r = subprocess.run([binary] + list(run_args),
                          capture_output=True, text=True, timeout=60,
                          cwd=build_dir)
        output = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return None, "binary timeout"
    except Exception as e:
        return None, f"binary run error: {e}"

    # 提取 kernel_time_us
    for line in output.split('\n'):
        if 'kernel_time_us' in line:
            try:
                return float(line.split('=')[1].strip()), None
            except:
                pass

    return None, f"no timing in output"


# ============================================================================
# Method 2: torch profiler 方式
# ============================================================================
def profile_via_torch_ops(op, cfg):
    """使用 torch.npu.profiler 获取 kernel 耗时"""
    import torch, torch_npu
    from torch_npu.profiler import profile, ProfilerActivity

    op_dir = find_op_dir(op)
    if not op_dir:
        return None, "op_dir not found"

    # 找 .so
    so_path = None
    for root, dirs, files in os.walk(op_dir):
        for f in files:
            if cfg.get("so_pattern", "") in f and f.endswith(".so"):
                so_path = os.path.join(root, f)
                break

    if not so_path:
        return None, "no .so found"

    try:
        torch.ops.load_library(so_path)
    except Exception as e:
        return None, f"load_library failed: {e}"

    # 找到 origin 模块和输入
    origin_mod_path = os.path.join(ORIGIN, f"{op}.py")
    if not os.path.isfile(origin_mod_path):
        return None, f"no origin module at {origin_mod_path}"

    sys.path.insert(0, ORIGIN)
    try:
        mod = __import__(op, fromlist=["Model", "get_inputs"])
        data = mod.get_inputs()
        model = mod.Model()
    except Exception as e:
        return None, f"import failed: {e}"

    # 准备输入
    if isinstance(data, (list, tuple)):
        npu_inputs = []
        for d in data:
            if isinstance(d, torch.Tensor):
                npu_inputs.append(d.npu())
            else:
                npu_inputs.append(d)
    else:
        npu_inputs = [data.npu() if isinstance(data, torch.Tensor) else data]

    # 找正确的 op 名称
    op_name = None
    for attr in dir(torch.ops.npu):
        if op.replace("_", "")[:5] in attr.replace("_", "")[:5]:
            op_name = attr
            break

    if not op_name:
        return None, "no matching op name found"

    op_fn = getattr(torch.ops.npu, op_name)

    # 使用 profiler 测量 kernel 时间
    outdir = tempfile.mkdtemp(prefix=f"prof_{op}_")
    activities = [ProfilerActivity.CPU, ProfilerActivity.NPU]

    try:
        # Warmup
        for _ in range(5):
            op_fn(*npu_inputs)
        torch.npu.synchronize()

        # Profiled run
        with profile(activities=activities, record_shapes=True) as prof:
            for _ in range(10):
                op_fn(*npu_inputs)
            torch.npu.synchronize()

        # Export and parse
        trace_path = os.path.join(outdir, "trace.json")
        prof.export_chrome_trace(trace_path)

        if os.path.isfile(trace_path):
            with open(trace_path) as f:
                trace = json.load(f)

            # 找 kernel 事件
            kernel_durs = []
            for evt in trace:
                cat = evt.get('cat', '')
                name = evt.get('name', '')
                dur = evt.get('dur', 0)
                # AscendC kernel 事件通常有特定 cat
                if cat in ('kernel', 'async_gpu') or 'kernel' in name.lower():
                    kernel_durs.append(dur)

            if kernel_durs:
                avg_kernel_us = sum(kernel_durs) / len(kernel_durs)
                return avg_kernel_us, None
            else:
                # 没有显式 kernel 事件，使用 cpu_op 时间估算
                cpu_durs = [e['dur'] for e in trace if e.get('cat') == 'cpu_op']
                if cpu_durs:
                    avg_us = sum(cpu_durs) / len(cpu_durs)
                    return avg_us, "no kernel events, using cpu_op time"

        return None, "no trace data"
    except Exception as e:
        return None, f"profiler error: {e}"
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


# ============================================================================
# Torch 参考时间测量 (纯 PyTorch kernel，用于对比)
# ============================================================================
def measure_torch_reference(op):
    """测量 PyTorch 参考实现耗时 (纯计算, 无 launch overhead)"""
    import torch, numpy as np

    origin_mod_path = os.path.join(ORIGIN, f"{op}.py")
    if not os.path.isfile(origin_mod_path):
        return None

    sys.path.insert(0, ORIGIN)
    try:
        mod = __import__(op, fromlist=["Model", "get_inputs"])
        data = mod.get_inputs()
        model = mod.Model()
    except:
        return None

    # CPU 上测量 (避开 NPU launch overhead)
    if isinstance(data, (list, tuple)):
        cpu_inputs = []
        for d in data:
            if isinstance(d, torch.Tensor):
                cpu_inputs.append(d.cpu().float())
            else:
                cpu_inputs.append(d)
    else:
        cpu_inputs = [data.cpu().float() if isinstance(data, torch.Tensor) else data]

    # 重复测量
    try:
        for _ in range(WARMUP):
            model.forward(*cpu_inputs)

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            model.forward(*cpu_inputs)
        elapsed = time.perf_counter() - t0
        return elapsed / REPEAT * 1e6  # us
    except Exception as e:
        return None


# ============================================================================
# Main
# ============================================================================
def main():
    os.chdir(MERGE)
    print("=" * 80)
    print(" 纯 Kernel 耗时测试 v2 (排除 launch overhead)")
    print(f" 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = OrderedDict()

    for op, cfg in OP_CONFIGS.items():
        print(f"\n{'─'*70}")
        print(f"[{op}] method={cfg['method']}", end=" ")

        op_dir = find_op_dir(op)
        if not op_dir:
            print("SKIP (op_dir not found)")
            continue

        kernel_us = None
        error = None
        method_used = None

        # ================================================================
        # Method 1: 注入计时 + 重编译 + 运行
        # ================================================================
        if cfg["method"] == 1:
            host_path = find_host_file(op_dir)

            if host_path:
                # 备份原文件
                backup = host_path + ".bak_kernel_timing"
                shutil.copy2(host_path, backup)

                try:
                    was_modified = inject_timing(host_path)
                    if was_modified:
                        print("injected,", end=" ")

                    # 编译
                    print("building...", end=" ", flush=True)
                    build_dir, build_err = build_with_run_sh(op_dir, op, skip_build=False)

                    if build_err:
                        print(f"BUILD FAIL: {build_err[:100]}")
                    else:
                        # 运行
                        run_args = cfg.get("run_args", [])
                        kernel_us, run_err = run_binary_and_extract(
                            op_dir, op, build_dir, run_args)

                        if kernel_us:
                            method_used = "host注入+重编译"
                            print(f"DONE kernel={kernel_us:.2f}us")
                        else:
                            print(f"RUN FAIL: {run_err[:100] if run_err else 'no timing'}")
                finally:
                    # 恢复原文件
                    shutil.copy2(backup, host_path)
                    os.remove(backup)
            else:
                print("no host file found, trying profiler...")
                cfg["method"] = 2  # fallback

        # ================================================================
        # Method 2: torch profiler
        # ================================================================
        if cfg["method"] == 2 and kernel_us is None:
            print("profiling...", end=" ", flush=True)
            try:
                kernel_us, prof_err = profile_via_torch_ops(op, cfg)
                if kernel_us:
                    method_used = "torch.profiler"
                    print(f"DONE kernel={kernel_us:.2f}us")
                else:
                    print(f"FAIL: {prof_err}")
                    error = prof_err
            except Exception as e:
                print(f"PROFILER ERROR: {e}")
                error = str(e)

        # ================================================================
        # Method 3: summary.json 兜底
        # ================================================================
        if cfg["method"] == 3 or (kernel_us is None and error):
            print("summary.json...", end=" ", flush=True)
            sf = None
            for root, dirs, files in os.walk(os.path.join(MERGE, op)):
                if "summary.json" in files:
                    sf = os.path.join(root, "summary.json")
                    break
            if sf:
                with open(sf) as f:
                    d = json.load(f)
                perf = d.get("perf_data", {})
                kernel_us = (perf.get("ascend_us") or perf.get("ascendc_kernel_us") or
                            perf.get("ascendc_us") or 0)
                if kernel_us and float(kernel_us) > 0:
                    method_used = "summary.json(hist)"
                    print(f"kernel={float(kernel_us):.2f}us")
                else:
                    print("no valid data")
                    kernel_us = None

        # ================================================================
        # 测量 Torch 参考时间
        # ================================================================
        torch_us = None
        print(f"  Torch ref...", end=" ", flush=True)
        torch_us = measure_torch_reference(op)
        if torch_us:
            print(f"{torch_us:.1f}us")
        else:
            print("SKIP (no reference)")

        # 保存结果
        if kernel_us and kernel_us > 0:
            ratio = torch_us / kernel_us if torch_us and torch_us > 0 else 0
            results[op] = {
                "kernel_time_us": round(float(kernel_us), 4),
                "torch_ref_us": round(torch_us, 4) if torch_us else None,
                "kernel_vs_torch_speedup": round(ratio, 4) if ratio else None,
                "method": method_used,
            }
        else:
            print(f"  => FAILED: {error or 'no kernel time'}")

    # ================================================================
    # 输出报告
    # ================================================================
    print("\n" + "=" * 80)
    print(" 纯 Kernel 耗时测试结果")
    print("=" * 80)
    print(f"{'算子':<26} {'Kernel(us)':>12} {'TorchCPU(us)':>14} {'Kernel加速比':>12} {'方法':<20}")
    print("-" * 84)

    for op, r in results.items():
        ku = r['kernel_time_us']
        tu = r.get('torch_ref_us') or 0
        speedup = r.get('kernel_vs_torch_speedup') or 0
        method = r.get('method', '?')
        flag = "🟢" if speedup > 1.0 else ("🟡" if speedup > 0.5 else "🔴")
        print(f"{flag} {op:<24} {ku:>12.2f} {tu:>14.1f} {speedup:>11.2f}x {method:<20}")

    # 统计
    n_total = len(OP_CONFIGS)
    n_ok = len(results)
    n_fast = sum(1 for r in results.values() if (r.get('kernel_vs_torch_speedup') or 0) > 1.0)
    n_slow = sum(1 for r in results.values() if 0 < (r.get('kernel_vs_torch_speedup') or 0) <= 1.0)

    print(f"\n成功: {n_ok}/{n_total} | 🟢Kernel加速: {n_fast} | 🔴Kernel减速: {n_slow}")

    # 保存
    out_path = os.path.join(MERGE, "benchmark_results", "kernel_time_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "纯 Kernel 耗时测试 (排除 launch overhead)",
            "total_ops": n_total,
            "successful_ops": n_ok,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    return results


if __name__ == "__main__":
    main()
