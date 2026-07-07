# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# hc_split_sinkhorn PyTorch 通路端到端测试
# ============================================================================

import numpy as np
import os
import struct
import sys
import torch
import torch_npu

from golden import compute_golden

# 加载自定义算子库
so_path = os.path.join(os.path.dirname(__file__), "..", "build", "libhc_split_sinkhorn_ops.so")
if os.path.exists(so_path):
    torch.ops.load_library(so_path)
else:
    print(f"WARNING: {so_path} not found, trying default path")
    torch.ops.load_library("libhc_split_sinkhorn_ops.so")


def load_meta(meta_path):
    max_hc = 32
    max_mix_hc = (2 + max_hc) * max_hc
    with open(meta_path, "rb") as f:
        data = f.read()
    b = struct.unpack_from("<Q", data, 0)[0]
    s = struct.unpack_from("<Q", data, 8)[0]
    hc = struct.unpack_from("<Q", data, 16)[0]
    # skip 2 reserved uint64
    iters = struct.unpack_from("<I", data, 40)[0]
    eps = struct.unpack_from("<f", data, 44)[0]

    offset = 48
    hc_scale = np.array([struct.unpack_from("<f", data, offset + i*4)[0] for i in range(3)], dtype=np.float32)

    offset += 12
    mix_hc = (2 + hc) * hc
    hc_base = np.array([struct.unpack_from("<f", data, offset + i*4)[0] for i in range(mix_hc)], dtype=np.float32)

    return b, s, hc, iters, eps, hc_scale, hc_base


def main():
    # 读取 meta 信息
    meta_path = "input/meta.bin"
    if not os.path.exists(meta_path):
        meta_path = os.path.join(os.path.dirname(__file__), "..", "build", "input", "meta.bin")
    if not os.path.exists(meta_path):
        print("ERROR: meta.bin not found")
        return False

    b, s, hc, iters, eps, hc_scale, hc_base = load_meta(meta_path)
    mix_hc = (2 + hc) * hc

    print(f"PyTorch test: b={b} s={s} hc={hc} iters={iters} eps={eps}")

    # 加载输入数据
    mixes_path = os.path.join(os.path.dirname(meta_path), "input_mixes.bin")
    mixes = np.fromfile(mixes_path, dtype=np.float32).reshape(b * s, mix_hc)

    # 计算 golden
    golden_pre, golden_post, golden_comb = compute_golden(mixes, hc, iters, eps, hc_scale, hc_base)
    B = b * s

    # PyTorch 张量准备
    mixes_t = torch.from_numpy(mixes.reshape(b, s, mix_hc)).npu()
    hc_scale_t = torch.from_numpy(hc_scale).npu()
    hc_base_t = torch.from_numpy(hc_base).npu()

    pre_t = torch.empty(b, s, hc, dtype=torch.float32).npu()
    post_t = torch.empty(b, s, hc, dtype=torch.float32).npu()
    comb_t = torch.empty(b, s, hc, hc, dtype=torch.float32).npu()

    # 调用算子
    torch.ops.npu.hc_split_sinkhorn(
        mixes_t, int(hc), int(iters), float(eps),
        hc_scale_t, hc_base_t,
        pre_t, post_t, comb_t)

    torch.npu.synchronize()

    # 精度验证
    pre_out = pre_t.cpu().numpy().reshape(B, hc)
    post_out = post_t.cpu().numpy().reshape(B, hc)
    comb_out = comb_t.cpu().numpy().reshape(B, hc, hc)

    all_pass = True

    for name, out, golden in [
        ("pre", pre_out, golden_pre),
        ("post", post_out, golden_post),
        ("comb", comb_out, golden_comb)]:

        abs_diff = np.abs(out - golden)
        max_abs = np.max(abs_diff)
        rel_diff = abs_diff / (np.abs(golden) + 1e-10)
        mere = np.mean(rel_diff)
        mare = np.max(rel_diff)
        pass_ = mere < 1.22e-4 and mare < 1.22e-3
        status = "PASS" if pass_ else "FAIL"
        print(f"  [{name}] {status}: MERE={mere:.2e} MARE={mare:.2e} max_abs_diff={max_abs:.2e}")
        if not pass_:
            all_pass = False

    if all_pass:
        print("PyTorch: ALL PASS")
    else:
        print("PyTorch: SOME FAILED")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
