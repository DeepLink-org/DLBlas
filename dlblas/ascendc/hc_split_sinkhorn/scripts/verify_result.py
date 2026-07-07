# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# hc_split_sinkhorn 精度验证（直调通路）
# 对比 kernel 输出 (output_pre.bin / output_post.bin / output_comb.bin)
#   与 golden  (golden_pre.bin / golden_post.bin / golden_comb.bin)
# ============================================================================

import numpy as np
import os
import sys
import struct
import glob


def verify_file(output_path, golden_path, name):
    if not os.path.exists(output_path):
        print(f"  [{name}] SKIP: output file not found ({output_path})")
        return True

    output = np.fromfile(output_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)

    if output.shape != golden.shape:
        print(f"  [{name}] FAIL: shape mismatch output={output.shape} golden={golden.shape}")
        return False

    if output.size == 0:
        print(f"  [{name}] PASS (empty)")
        return True

    abs_diff = np.abs(output - golden)
    max_abs = np.max(abs_diff)
    rel_diff = abs_diff / (np.abs(golden) + 1e-10)
    mere = np.mean(rel_diff)
    mare = np.max(rel_diff)

    # 精度标准: MERE < 1.22e-4, MARE < 1.22e-3
    mere_pass = mere < 1.22e-4
    mare_pass = mare < 1.22e-3

    status = "PASS" if (mere_pass and mare_pass) else "FAIL"
    print(f"  [{name}] {status}: MERE={mere:.2e} MARE={mare:.2e} max_abs_diff={max_abs:.2e}")

    return mere_pass and mare_pass


def main():
    # 读取 meta.bin 获取 shape 信息
    max_hc = 32
    max_mix_hc = (2 + max_hc) * max_hc

    meta_path = "input/meta.bin"
    if not os.path.exists(meta_path):
        # 尝试从当前目录的上级找
        meta_path = "../input/meta.bin"
    if not os.path.exists(meta_path):
        print("WARNING: meta.bin not found, skipping shape check")
        b, s, hc = 0, 0, 0
    else:
        with open(meta_path, "rb") as f:
            data = f.read()
        b = struct.unpack_from("<Q", data, 0)[0]
        s = struct.unpack_from("<Q", data, 8)[0]
        hc = struct.unpack_from("<Q", data, 16)[0]

    totalBatch = b * s
    print(f"Verifying: b={b} s={s} hc={hc} totalBatch={totalBatch}")

    all_pass = True

    # 查找 output 和 golden 目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        output_dir = "../output"
    golden_dir = "output"
    if not os.path.exists(golden_dir):
        golden_dir = "../output"

    # 验证 pre
    all_pass &= verify_file(
        os.path.join(output_dir, "output_pre.bin"),
        os.path.join(golden_dir, "golden_pre.bin"),
        "pre")
    # 验证 post
    all_pass &= verify_file(
        os.path.join(output_dir, "output_post.bin"),
        os.path.join(golden_dir, "golden_post.bin"),
        "post")
    # 验证 comb
    all_pass &= verify_file(
        os.path.join(output_dir, "output_comb.bin"),
        os.path.join(golden_dir, "golden_comb.bin"),
        "comb")

    if all_pass:
        print("Overall: ALL PASS")
        return 0
    else:
        print("Overall: SOME FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
