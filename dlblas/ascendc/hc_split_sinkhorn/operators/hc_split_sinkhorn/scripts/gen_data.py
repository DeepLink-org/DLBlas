# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# hc_split_sinkhorn 测试数据生成
#
# 用法: python3 gen_data.py <b> <s> <hc> <sinkhorn_iters> <eps> [seed]
#   默认: python3 gen_data.py 2 8 4 20 1e-6 0
# ============================================================================

import numpy as np
import os
import struct
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 解析参数
b = int(sys.argv[1]) if len(sys.argv) > 1 else 2
s = int(sys.argv[2]) if len(sys.argv) > 2 else 8
hc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
sinkhorn_iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
eps = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0

np.random.seed(seed)

mix_hc = (2 + hc) * hc
totalBatch = b * s
dtype = np.float32

print(f"Generating data: b={b} s={s} hc={hc} totalBatch={totalBatch} mix_hc={mix_hc} iters={sinkhorn_iters} eps={eps}")

# 生成输入数据
mixes = np.random.randn(totalBatch, mix_hc).astype(dtype) * 0.5
hc_scale = np.array([1.0, 1.0, 0.5], dtype=dtype)
hc_base = np.random.randn(mix_hc).astype(dtype) * 0.1

# 写出 mixes 到二进制文件
mixes.tofile("input/input_mixes.bin")

# 写出 meta.bin: 8*uint64(b, s, hc, reserved, reserved) + uint32(iters) + float(eps) + 3*float(hcScale) + MAX_MIX_HC*float(hcBase)
MAX_HC = 32
MAX_MIX_HC = (2 + MAX_HC) * MAX_HC

with open("input/meta.bin", "wb") as f:
    f.write(struct.pack("<Q", b))
    f.write(struct.pack("<Q", s))
    f.write(struct.pack("<Q", hc))
    f.write(struct.pack("<Q", 0))  # reserved
    f.write(struct.pack("<Q", 0))  # reserved
    f.write(struct.pack("<I", sinkhorn_iters))
    f.write(struct.pack("<f", eps))
    for i in range(3):
        f.write(struct.pack("<f", float(hc_scale[i])))
    # hcBase: 填充到 MAX_MIX_HC
    for i in range(mix_hc):
        f.write(struct.pack("<f", float(hc_base[i])))
    for i in range(mix_hc, MAX_MIX_HC):
        f.write(struct.pack("<f", 0.0))

# 计算 golden
pre, post, comb = compute_golden(mixes, hc, sinkhorn_iters, eps, hc_scale, hc_base)

# 写出 golden 到二进制文件
pre.astype(dtype).tofile("output/golden_pre.bin")
post.astype(dtype).tofile("output/golden_post.bin")
comb.astype(dtype).tofile("output/golden_comb.bin")

print(f"Data generated successfully.")
print(f"  mixes: {mixes.shape} {mixes.dtype}")
print(f"  pre:   {pre.shape} {pre.dtype}")
print(f"  post:  {post.shape} {post.dtype}")
print(f"  comb:  {comb.shape} {comb.dtype}")
