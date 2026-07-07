#!/usr/bin/env python3
"""
自动给每个算子 host 代码注入 kernel 计时, 重编译, 运行, 收集纯 kernel 耗时
用法: python3 bench_kernel_only.py
"""
import os, sys, re, subprocess, time, json

MERGE = "/mnt/data01/zmz/workspace/12agent/waic/cannbot-merge"
ORIGIN = "/mnt/data01/zmz/workspace/12agent/waic/origin"
sys.path.insert(0, ORIGIN)

# 需要测试的算子 (有 host + 可编译二进制的)
OPS = [
    "engram_gate_bwd",   # 已有计时, 直接跑
    "expand_kenel_fwd", "apply_mix", "hc_split_sinkhorn",
    "head_compute_mix_fwd", "head_compute_mix_bwd",
    "big_fuse", "mhc_post", "norm_fn", "pre_split_mixes",
    "engram_gate_fwd", "engram_gate_w_reduce",
    "expand_kenel_bwd", "MTPBlock", "sparse_attn",
    "engram_fused_weight", "act_quant_kernel",
]

def find_host(op_dir):
    for root, dirs, files in os.walk(op_dir):
        for f in files:
            if f.endswith(".asc") and "kernel" not in f and "diag" not in f and "data_utils" not in f:
                return os.path.join(root, f), f
    return None, None

def find_build_dir(op_dir):
    """找 CMakeLists.txt 所在的 build 目录"""
    for root, dirs, files in os.walk(op_dir):
        if "CMakeLists.txt" in files and "build" in root.split(os.sep):
            return root
        if "CMakeLists.txt" in files:
            # 这是源码目录，build 在旁边
            src_dir = root
            build_dir = os.path.join(src_dir, "build")
            if not os.path.isdir(build_dir):
                build_dir = os.path.join(os.path.dirname(src_dir), "build")
            return build_dir if os.path.isdir(build_dir) else None
    return None

def inject_timing(host_path):
    """给 host 文件注入 kernel 计时, 返回是否修改"""
    with open(host_path) as f:
        content = f.read()

    # 检查是否已有计时
    if "kernel_time_us" in content:
        return False  # 已有

    # 找 kernel launch 行: _kernel<<<...>>> 或 KernelCall
    kernel_patterns = [
        r'(\w+_kernel<<<.*?>>>\s*\()',
        r'(KernelCall\s*\()',
    ]

    modified = False

    # 添加 #include <chrono>
    if "#include <chrono>" not in content and "<chrono>" not in content:
        # 在最后一个 #include 之后插入
        last_include = content.rfind("#include")
        end_of_line = content.find("\n", last_include)
        content = (content[:end_of_line+1] +
                   '#include <chrono>\n' +
                   content[end_of_line+1:])
        modified = True

    # 找 kernel launch 并包装计时
    lines = content.split('\n')
    new_lines = []
    kernel_found = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 检测 kernel launch
        is_kernel = ('_kernel<<<' in stripped or 'KernelCall(' in stripped) and not stripped.startswith('//')

        if is_kernel and not kernel_found:
            kernel_found = True
            # 缩进
            indent = line[:len(line) - len(line.lstrip())]

            # 在 kernel launch 前加入: sync + t0
            new_lines.append(f'{indent}aclrtSynchronizeStream(stream);')
            new_lines.append(f'{indent}auto _t0 = std::chrono::high_resolution_clock::now();')
            new_lines.append(line)  # kernel launch

            # 找下一行的 aclrtSynchronizeStream 替换为计时版
            # 在 kernel launch 后的 aclrtSynchronizeStream 之后加 t1 + printf
            # 继续扫描后续行
            modified = True
        elif kernel_found and 'aclrtSynchronizeStream' in stripped:
            # 替换这个 sync: 在它之后加计时
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(line)  # 保留原 sync
            new_lines.append(f'{indent}auto _t1 = std::chrono::high_resolution_clock::now();')
            new_lines.append(f'{indent}double _kernel_us = std::chrono::duration<double, std::micro>(_t1 - _t0).count();')
            new_lines.append(f'{indent}printf("[PERF] kernel_time_us=%.2f\\n", _kernel_us);')
            kernel_found = False  # reset
        else:
            new_lines.append(line)

    if modified:
        with open(host_path, 'w') as f:
            f.write('\n'.join(new_lines))
    return modified

def build_and_run(op, op_dir, build_dir):
    """编译并运行, 返回 kernel 耗时"""
    os.chdir(build_dir)

    # cmake
    r = subprocess.run(["cmake", ".."], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        # cmake 可能已经在不同目录, 尝试找到正确的源码目录
        for src_dir in [os.path.dirname(build_dir), os.path.join(os.path.dirname(build_dir), "..")]:
            os.chdir(build_dir)
            r = subprocess.run(["cmake", src_dir], capture_output=True, text=True, timeout=60)
            if r.returncode == 0: break

    # make
    r = subprocess.run(["make", "-j4"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, f"make failed: {r.stderr[-200:]}"

    # 找二进制
    binary = os.path.join(build_dir, op)
    if not os.path.exists(binary):
        # 找子目录
        for root, dirs, files in os.walk(build_dir):
            if op in files and os.access(os.path.join(root, op), os.X_OK):
                binary = os.path.join(root, op); break

    if not os.path.exists(binary):
        return None, "binary not found"

    # 运行
    try:
        r = subprocess.run([binary], capture_output=True, text=True, timeout=30)
    except:
        # 有些需要参数
        try:
            r = subprocess.run([binary, "14", "4", "128"], capture_output=True, text=True, timeout=30)
        except:
            return None, f"binary run failed"

    # 提取计时
    output = r.stdout + r.stderr
    for line in output.split('\n'):
        if 'kernel_time_us' in line:
            try:
                return float(line.split('=')[1].strip()), None
            except: pass

    return None, f"no timing in output (first 200 chars: {output[:200]})"

# ============ main ============
results = {}
os.chdir(MERGE)

print("=" * 70)
print(" 纯 Kernel 耗时测试 (排除 launch overhead)")
print("=" * 70)

for op in OPS:
    print(f"\n[{op}]", end=" ")

    op_dir = os.path.join(MERGE, op)
    if not os.path.isdir(op_dir):
        print("SKIP (dir not found)")
        continue

    host_path, host_name = find_host(op_dir)
    if not host_path:
        print("SKIP (no host file)")
        continue

    # 注入计时
    was_modified = inject_timing(host_path)
    if was_modified:
        print("injected timing,", end=" ")

    # 找 build 目录
    build_dir = find_build_dir(op_dir)
    if not build_dir:
        print("SKIP (no build dir)")
        continue

    print(f"building...", end=" ", flush=True)

    kernel_us, err = build_and_run(op, op_dir, build_dir)

    if kernel_us:
        results[op] = kernel_us
        print(f"kernel_time={kernel_us:.2f}us")
    else:
        print(f"FAIL: {err}")

# ============ 输出 ============
print("\n" + "=" * 70)
print(" 纯 Kernel 耗时汇总")
print("=" * 70)
print(f"{'算子':<26} {'Kernel(us)':>12}")
print("-" * 40)
for op, t in sorted(results.items()):
    print(f"  {op:<24} {t:>12.2f}")

# 对已有的 engram_gate_bwd 作为基准
if "engram_gate_bwd" in results:
    base = results["engram_gate_bwd"]
    print(f"\n基准 (engram_gate_bwd): {base:.2f} us")

print(f"\n成功: {len(results)}/{len(OPS)} 个算子")
