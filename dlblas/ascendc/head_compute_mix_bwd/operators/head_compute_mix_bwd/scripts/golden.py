# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Golden computation for head_compute_mix_bwd operator
# ============================================================================

import numpy as np


def compute_golden(input_mix, mhc_scale, mhc_base, grad_out):
    """Compute the reference output for head_compute_mix_bwd.

    Args:
        input_mix:  (B, S, C) numpy array or torch.Tensor
        mhc_scale:  (1,) scalar
        mhc_base:   (C,) per-channel bias
        grad_out:   (B, S, C) upstream gradient

    Returns:
        (grad_input_mix, grad_mhc_scale, grad_mhc_base)
    """
    z = input_mix * mhc_scale + mhc_base
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    grad_z = grad_out * sigmoid * (1.0 - sigmoid)
    grad_input_mix = grad_z * mhc_scale
    grad_mhc_base = grad_z.sum(axis=(0, 1), keepdims=True).reshape(-1)
    grad_mhc_scale = (grad_z * input_mix).sum(axis=(0, 1, 2), keepdims=True).reshape(1)
    return grad_input_mix, grad_mhc_scale, grad_mhc_base
