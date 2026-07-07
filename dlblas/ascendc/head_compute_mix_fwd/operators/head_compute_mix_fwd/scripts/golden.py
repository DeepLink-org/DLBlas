# ============================================================================
# Golden computation for head_compute_mix_fwd
# Shared by gen_data.py and test_torch.py
#
# Computes: output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
# ============================================================================

import numpy as np


def compute_golden(input_mix, mhc_scale, mhc_base, mhc_pre_eps):
    """
    Compute reference output for head_compute_mix_fwd.

    Args:
        input_mix: numpy array or torch.Tensor, FP16, shape [batch, n1, mhc_mult]
        mhc_scale: numpy array or torch.Tensor, FP16, scalar
        mhc_base: numpy array or torch.Tensor, FP16, [mhc_mult]
        mhc_pre_eps: float (FP32)

    Returns:
        FP16 numpy array with same shape as input_mix
    """
    # Detect if inputs are torch tensors
    try:
        import torch
        is_torch = isinstance(input_mix, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        # PyTorch pathway: use FP32 for reference computation
        x = input_mix.float() * mhc_scale.float() + mhc_base.float()
        result = torch.sigmoid(x) + mhc_pre_eps
        return result.half()
    else:
        # NumPy pathway
        x = input_mix.astype(np.float32) * mhc_scale.astype(np.float32) + mhc_base.astype(np.float32)
        # sigmoid(x) = 1 / (1 + exp(-x))
        result = 1.0 / (1.0 + np.exp(-x)) + mhc_pre_eps
        return result.astype(np.float16)
