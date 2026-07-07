# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Golden 计算 - Expand Kernel Backward 算子 (双通路共用)
# 语义: o_grad.sum(dim=-2)
# 使用 FP32 累加确保与 Ascend C Add API 内部升精度行为一致
# ============================================================================

import numpy as np
import torch


def compute_golden(o_grad):
    """计算算子的参考输出: sum along dim=-2 (mhc_mult 维度).

    使用 FP32 累加以匹配 Ascend C Add API 的内部精度提升行为。

    Args:
        o_grad: numpy array or torch.Tensor, shape (n0, n1, mhc_mult, h)

    Returns:
        与输入同类型的参考输出, shape (n0, n1, h)
    """
    if isinstance(o_grad, np.ndarray):
        # numpy: 升到 FP32 累加，再截断回 FP16
        return o_grad.astype(np.float32).sum(axis=-2).astype(np.float16)
    else:
        # torch: 升到 FP32 累加，再截断回原类型
        return o_grad.float().sum(dim=-2).to(o_grad.dtype)
