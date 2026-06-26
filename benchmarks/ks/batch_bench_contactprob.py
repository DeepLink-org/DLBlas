import subprocess
import re
import time

# ===================== 配置区域 =====================
TRIAL_COUNT = 50  # 跄1�750欄1�7
V0_FILE = "//workspace/DLBlas/dlblas/kernels/ks_competition/triton/layernorm_v1.py"
V1_FILE = "/workspace/DLBlas/dlblas/kernels/ks_competition/triton/layernorm_v2.py"
BENCH_SCRIPT = "/workspace/DLBlas/benchmarks/ks/auto_bench.py"
# ====================================================

v0_time_list = []
v1_time_list = []
speedup_list = []
pass_count = 0
fail_count = 0

print(f"弢�始批量评测，总轮数：{TRIAL_COUNT}")
print(f"v0: {V0_FILE}")
print(f"v1: {V1_FILE}\n")

for idx in range(TRIAL_COUNT):
    cmd = [
        "python", BENCH_SCRIPT,
        "--v0_file", V0_FILE,
        "--v1_file", V1_FILE
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stdout = result.stdout
        stderr = result.stderr

        if "PASS accuracy" in stdout:
            pass_count += 1
            # 正则提取耗时与加速比
            pat = re.compile(r"v0=([\d\.]+)\s*ms,\s*v1=([\d\.]+)\s*ms,\s*speedup=([\d\.]+)x")
            match = pat.search(stdout)
            if match:
                t0 = float(match.group(1))
                t1 = float(match.group(2))
                sp = float(match.group(3))
                v0_time_list.append(t0)
                v1_time_list.append(t1)
                speedup_list.append(sp)
                print(f"[{idx+1:2d}/{TRIAL_COUNT}] PASS | v0={t0:.6f}ms v1={t1:.6f}ms speedup={sp:.4f}")
            else:
                print(f"[{idx+1:2d}/{TRIAL_COUNT}] PASS 但解析��时失败")
        else:
            fail_count += 1
            print(f"[{idx+1:2d}/{TRIAL_COUNT}] FAIL")
            print("stdout:\n", stdout)
            print("stderr:\n", stderr)

    except Exception as e:
        fail_count += 1
        print(f"[{idx+1:2d}/{TRIAL_COUNT}] 执行异常: {e}")

# 统计汇��1�7
print("\n==================== 统计汇��1�7 ====================")
print(f"总次敄1�7: {TRIAL_COUNT} | 通过: {pass_count} | 失败: {fail_count}")
if len(v0_time_list) > 0:
    avg_v0 = sum(v0_time_list) / len(v0_time_list)
    avg_v1 = sum(v1_time_list) / len(v1_time_list)
    avg_sp = sum(speedup_list) / len(speedup_list)

    print(f"v0 平均耗时: {avg_v0:.6f} ms")
    print(f"v1 平均耗时: {avg_v1:.6f} ms")
    print(f"平均加��比: {avg_sp:.4f} x")
else:
    print("无有效成功数捄1�7")