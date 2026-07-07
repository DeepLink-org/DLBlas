# Sinkhorn Normalize - Golden 计算（双通路共用）

import numpy as np


def softmax_np(x, axis=-1):
    """NumPy softmax (数值稳定版本)"""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_sub = x - x_max
    x_exp = np.exp(x_sub)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum


def compute_golden(x, mhc=4, repeat=10, eps=1e-6):
    """计算 Sinkhorn Normalize 的参考输出。

    Args:
        x: numpy array, shape [1, batch, mhc, mhc]
        mhc: 矩阵维度
        repeat: 迭代次数
        eps: epsilon 防止除零

    Returns:
        numpy array, 与输入同 shape
    """
    # Step 0: Softmax(dim=-1) + eps
    y = softmax_np(x, axis=-1) + eps

    # Step 1: 列归一化 (sum dim=-2)
    col_sum = np.sum(y, axis=-2, keepdims=True) + eps
    y = y / col_sum

    # Step 2..repeat: 迭代行归一化 + 列归一化
    for _ in range(repeat - 1):
        # 行归一化 (sum dim=-1)
        row_sum = np.sum(y, axis=-1, keepdims=True) + eps
        y = y / row_sum

        # 列归一化 (sum dim=-2)
        col_sum = np.sum(y, axis=-2, keepdims=True) + eps
        y = y / col_sum

    return y
