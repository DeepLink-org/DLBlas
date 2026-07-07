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
# Golden 计算（双通路共用）
# 实现 pre_split_mixes 的参考计算逻辑
# ============================================================================

import numpy as np


def compute_golden(input_mixes, mhc_scale, mhc_base, mhc_mult, mhc_pre_eps, mhc_post_mult_value):
    """计算 pre_split_mixes 的参考输出。

    Args:
        input_mixes:  numpy array, shape [batch, seq_len, M3], dtype float32
        mhc_scale:    numpy array, shape [3], dtype float32
        mhc_base:     numpy array, shape [M3], dtype float32
        mhc_mult:     int, m 值
        mhc_pre_eps:  float
        mhc_post_mult_value: float

    Returns:
        (pre_mix, post_mix, comb_mix): tuple of numpy arrays
          pre_mix:  [batch, seq_len, m] float32
          post_mix: [batch, seq_len, m] float32
          comb_mix: [batch, seq_len, m*m] float32
    """
    m = mhc_mult
    a, b = input_mixes.shape[:2]
    scale = np.concatenate([
        np.full(m, mhc_scale[0], dtype=np.float32),
        np.full(m, mhc_scale[1], dtype=np.float32),
        np.full(m * m, mhc_scale[2], dtype=np.float32),
    ])
    x = input_mixes * scale + mhc_base

    pre = x[:, :, :m]
    pre_sig = 1.0 / (1.0 + np.exp(-pre))
    pre_mix = pre_sig + mhc_pre_eps
    # 展平 pre: [batch, seq_len, m] → [batch, seq_len, m]
    pre_mix = pre_mix.reshape(a, b, m)

    post = x[:, :, m:2 * m]
    post_sig = 1.0 / (1.0 + np.exp(-post))
    post_mix = post_sig * mhc_post_mult_value
    post_mix = post_mix.reshape(a, b, m)

    comb = x[:, :, 2 * m:]
    comb_mix = comb.reshape(a, b, m * m)

    return pre_mix, post_mix, comb_mix
