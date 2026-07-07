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
# Golden 计算（双通路共用：直调 & PyTorch）
# ============================================================================

import numpy as np


def compute_golden(x, mhc_mult):
    """计算算子的参考输出。

    等价于: x.unsqueeze(-2).expand(..., mhc_mult, x.shape[-1]).contiguous()

    Args:
        x: numpy array 或 torch.Tensor, shape (..., H)
        mhc_mult: 扩展倍数

    Returns:
        与输入同类型的参考输出, shape (..., mhc_mult, H)
    """
    # torch.Tensor 输入
    if hasattr(x, 'cpu'):
        x_cpu = x.cpu()
        # 使用 PyTorch 操作（支持所有 dtype 包括 BF16）
        out = x_cpu.unsqueeze(-2).expand(
            *x_cpu.shape[:-1], mhc_mult, x_cpu.shape[-1]
        ).contiguous()
        return out  # 返回 torch.Tensor，保留原始 dtype (BF16 etc.)
    else:
        # numpy array 输入
        x_np = np.asarray(x)
        expanded = np.expand_dims(x_np, axis=-2)
        repeat_shape = [1] * expanded.ndim
        repeat_shape[-2] = mhc_mult
        result = np.tile(expanded, repeat_shape)
        return np.ascontiguousarray(result)
