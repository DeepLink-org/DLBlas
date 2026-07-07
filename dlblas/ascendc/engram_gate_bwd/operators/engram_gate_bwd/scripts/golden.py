# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Golden reference computation for engram_gate_bwd
# ============================================================================

import numpy as np


def compute_golden(grad_out_bf16, x_bf16, k_bf16, v_bf16, wh_bf16, we_bf16, clamp_value=1e-6, eps=1e-20):
    """Compute engram_gate_bwd reference output (f32 golden).

    Args:
        grad_out_bf16: (T, H, D) numpy array, bf16
        x_bf16: (T, H, D) numpy array, bf16
        k_bf16: (T, H, D) numpy array, bf16
        v_bf16: (T, D) numpy array, bf16
        wh_bf16: (H, D) numpy array, bf16
        we_bf16: (H, D) numpy array, bf16
        clamp_value: float
        eps: float

    Returns:
        grad_x, grad_k, grad_v, grad_wh, grad_we (all f32)
    """
    x  = x_bf16.astype(np.float32)
    k  = k_bf16.astype(np.float32)
    v  = v_bf16.astype(np.float32)
    wh = wh_bf16.astype(np.float32)
    we = we_bf16.astype(np.float32)
    go = grad_out_bf16.astype(np.float32)

    D = x.shape[-1]
    scalar = D ** -0.5

    # Phase A: Forward recompute
    rstd_x  = 1.0 / np.sqrt(np.mean(x ** 2, axis=-1) + eps)
    rstd_k  = 1.0 / np.sqrt(np.mean(k ** 2, axis=-1) + eps)
    raw_dot = np.sum(x * wh * (k * we), axis=-1)
    dot     = raw_dot * rstd_x * rstd_k * scalar
    abs_dot = np.abs(dot)
    s_sqrt  = np.sign(dot) * np.sqrt(np.maximum(abs_dot, clamp_value))
    gate    = 1.0 / (1.0 + np.exp(-s_sqrt))

    # Phase B: Backward
    grad_v = np.sum(go * gate[:, :, np.newaxis], axis=1)  # (T, D)
    grad_gate = np.sum(go * v[:, np.newaxis, :], axis=-1)  # (T, H)
    grad_s_sqrt = grad_gate * gate * (1.0 - gate)
    mask = (abs_dot >= clamp_value).astype(np.float32)
    grad_dot = grad_s_sqrt * mask * 0.5 / np.sqrt(np.maximum(abs_dot, clamp_value))
    grad_raw_dot = grad_dot * rstd_x * rstd_k * scalar
    grad_rstd_x  = grad_dot * raw_dot * rstd_k * scalar
    grad_rstd_k  = grad_dot * raw_dot * rstd_x * scalar

    # grad_x
    gx = (go
          + grad_raw_dot[:, :, np.newaxis] * wh * (k * we)
          + grad_rstd_x[:, :, np.newaxis] * (-x / D) * (rstd_x ** 3)[:, :, np.newaxis])
    # grad_k
    gk = (grad_raw_dot[:, :, np.newaxis] * we * (x * wh)
          + grad_rstd_k[:, :, np.newaxis] * (-k / D) * (rstd_k ** 3)[:, :, np.newaxis])
    # grad_wh, grad_we
    gwh = np.sum(grad_raw_dot[:, :, np.newaxis] * (k * we) * x, axis=0)
    gwe = np.sum(grad_raw_dot[:, :, np.newaxis] * (x * wh) * k, axis=0)

    return (gx.astype(np.float32), gk.astype(np.float32),
            grad_v.astype(np.float32), gwh.astype(np.float32), gwe.astype(np.float32))
