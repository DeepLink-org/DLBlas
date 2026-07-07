# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 测试数据生成脚本
# 用法：python3 gen_data.py [case_id]
#   case_id: "T1", "T2", "T3", ... (默认 "T2")
# ============================================================================

import numpy as np
import os
import sys
import struct

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 测试用例定义
TEST_CASES = {
    "T1": {"batch": 1, "seq_len": 1,    "m": 4,  "desc": "极小 shape"},
    "T2": {"batch": 1, "seq_len": 1024, "m": 4,  "desc": "基准"},
    "T3": {"batch": 8, "seq_len": 512,  "m": 4,  "desc": "大 batch, 多核"},
    "T4": {"batch": 1, "seq_len": 2048, "m": 4,  "desc": "大 seq_len"},
    "T5": {"batch": 1, "seq_len": 1024, "m": 1,  "desc": "m=1 边界"},
    "T6": {"batch": 1, "seq_len": 1024, "m": 8,  "desc": "m=8"},
    "T7": {"batch": 1, "seq_len": 1024, "m": 16, "desc": "m=16"},
    "T8": {"batch": 2, "seq_len": 256,  "m": 4,  "desc": "小 batch x 小 seq_len"},
}

# 默认参数
DEFAULT_EPS = 1e-2
DEFAULT_POST_MULT = 2.0

case_id = sys.argv[1] if len(sys.argv) > 1 else "T2"
if case_id not in TEST_CASES:
    print(f"Unknown test case: {case_id}. Available: {list(TEST_CASES.keys())}")
    sys.exit(1)

tc = TEST_CASES[case_id]
batch   = tc["batch"]
seq_len = tc["seq_len"]
m       = tc["m"]

M3 = 2 * m + m * m
totalRows = batch * seq_len
dtype = np.float32

print(f"Test case {case_id}: {tc['desc']}")
print(f"  batch={batch}, seq_len={seq_len}, m={m}, M3={M3}, totalRows={totalRows}")

# -- 生成随机权重 --
np.random.seed(42)
mhc_scale = (np.random.randn(3) * 0.1).astype(dtype)
mhc_base  = (np.random.randn(M3) * 0.1).astype(dtype)

# -- 生成随机输入 --
input_mixes = np.random.randn(batch, seq_len, M3).astype(dtype)

# -- 计算 Golden --
pre_mix, post_mix, comb_mix = compute_golden(
    input_mixes, mhc_scale, mhc_base, m, DEFAULT_EPS, DEFAULT_POST_MULT
)

# -- 写入 binary 文件 --
# 输入
input_mixes.tofile("input/input_mixes.bin")
mhc_scale.tofile("input/mhc_scale.bin")
mhc_base.tofile("input/mhc_base.bin")

# tiling_config: int64 totalRows, int32 m, float eps, float postMult
config_data = struct.pack("=qiff", totalRows, m, DEFAULT_EPS, DEFAULT_POST_MULT)
with open("input/tiling_config.bin", "wb") as f:
    f.write(config_data)

# 输出 (golden)
pre_mix.tofile("output/pre_mix.bin")
post_mix.tofile("output/post_mix.bin")
comb_mix.tofile("output/comb_mix.bin")

print(f"Input shapes:")
print(f"  input_mixes: {input_mixes.shape} ({input_mixes.nbytes} B)")
print(f"  mhc_scale:   {mhc_scale.shape} ({mhc_scale.nbytes} B)")
print(f"  mhc_base:    {mhc_base.shape} ({mhc_base.nbytes} B)")
print(f"Output shapes:")
print(f"  pre_mix:     {pre_mix.shape} ({pre_mix.nbytes} B)")
print(f"  post_mix:    {post_mix.shape} ({post_mix.nbytes} B)")
print(f"  comb_mix:    {comb_mix.shape} ({comb_mix.nbytes} B)")
print(f"Done.")
