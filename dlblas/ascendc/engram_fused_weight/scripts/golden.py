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
# Golden computation for engram_fused_weight
#
# Math: output = wh_data * we_data  (both FP32 after BF16→FP32 cast)
#
# Shared by gen_data.py (direct invoke) and test_torch.py (PyTorch path).
# ============================================================================

import numpy as np


def compute_golden(wh_data, we_data):
    """Compute reference output: wh * we (elementwise multiply).

    Args:
        wh_data: numpy array or torch.Tensor (FP32)
        we_data: numpy array or torch.Tensor (FP32)

    Returns:
        Elementwise product in FP32
    """
    return wh_data * we_data
