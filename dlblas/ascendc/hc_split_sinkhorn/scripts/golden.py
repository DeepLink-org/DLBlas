# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# hc_split_sinkhorn Golden 计算（双通路共用：gen_data.py + test_torch.py）
# ============================================================================

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_golden(mixes, hc_mult, sinkhorn_iters, eps, hc_scale, hc_base):
    """计算 hc_split_sinkhorn 算子的参考输出。

    Args:
        mixes:       numpy array, shape (B, mix_hc), dtype float32
        hc_mult:     int, hc 维度大小
        sinkhorn_iters: int, Sinkhorn 迭代次数
        eps:         float, 数值稳定常数
        hc_scale:    numpy array, shape (3,)
        hc_base:     numpy array, shape (mix_hc,)

    Returns:
        (pre, post, comb) 三元组，均为 numpy array, dtype float32
        pre:  shape (B, hc)
        post: shape (B, hc)
        comb: shape (B, hc, hc)
    """
    B = mixes.shape[0]
    hc = hc_mult
    mix_hc = (2 + hc) * hc

    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

    # 拆分
    pre_raw = mixes[:, :hc]
    post_raw = mixes[:, hc:2*hc]
    comb_raw = mixes[:, 2*hc:2*hc+hc*hc].reshape(B, hc, hc)

    base_pre = hc_base[:hc]
    base_post = hc_base[hc:2*hc]
    base_comb = hc_base[2*hc:2*hc+hc*hc].reshape(1, hc, hc)

    # Pre: sigmoid(x * s0 + bias_pre) + eps
    pre = sigmoid(pre_raw * s0 + base_pre) + eps

    # Post: 2 * sigmoid(x * s1 + bias_post)
    post = 2.0 * sigmoid(post_raw * s1 + base_post)

    # Comb: Sinkhorn
    comb = comb_raw * s2 + base_comb  # (B, hc, hc)

    # 第 0 次迭代: exp 稳定化
    for b in range(B):
        mat = comb[b]
        # 行最大值稳定化
        for r in range(hc):
            row_max = np.max(mat[r, :])
            mat[r, :] = np.exp(mat[r, :] - row_max)
        # 行归一化 (+eps)
        for r in range(hc):
            row_sum = np.sum(mat[r, :])
            mat[r, :] = mat[r, :] / row_sum + eps
        # 列归一化
        for c in range(hc):
            col_sum = np.sum(mat[:, c])
            mat[:, c] = mat[:, c] / (col_sum + eps)

    # 第 1..sinkhorn_iters-1 次迭代
    for _ in range(1, sinkhorn_iters):
        for b in range(B):
            mat = comb[b]
            for r in range(hc):
                row_sum = np.sum(mat[r, :])
                mat[r, :] = mat[r, :] / (row_sum + eps)
            for c in range(hc):
                col_sum = np.sum(mat[:, c])
                mat[:, c] = mat[:, c] / (col_sum + eps)

    return pre, post, comb
