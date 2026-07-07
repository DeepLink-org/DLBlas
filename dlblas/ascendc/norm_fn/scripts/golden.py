# Golden 计算 (numpy 参考实现)
# 被 gen_data.py 和 verify_result.py 共同引用

import numpy as np


def compute_golden_torch(residual_f32, mhc_fn, mhc_norm_weight, mhc_norm_eps):
    """计算 norm_fn 参考输出 (PyTorch 通路，输入为 float32).

    Args:
        residual_f32: numpy array, shape (1, 13, 4, 1280), dtype float32
        mhc_fn: numpy array, shape (24, 5120), dtype float32
        mhc_norm_weight: numpy array or None, shape (5120,), dtype float32
        mhc_norm_eps: float

    Returns:
        numpy array, shape (1, 13, 24), dtype float32
    """
    mhc_fn_f32 = mhc_fn.astype(np.float32)
    residual = residual_f32.astype(np.float32)

    if mhc_norm_weight is not None:
        mhc_fn_f32 = mhc_fn_f32 * mhc_norm_weight.astype(np.float32)

    residual_flat = residual.reshape(1, 13, -1)
    mhc_mult = mhc_fn_f32.shape[0]
    rms_group_size = mhc_fn_f32.shape[-1]

    residual_view = residual_flat.reshape(-1, 1, rms_group_size)
    mhc_fn_view = mhc_fn_f32.reshape(mhc_mult, 1, rms_group_size)

    mixes = np.einsum('mbk,nbk->mbn', residual_view, mhc_fn_view)
    sqrsum = (residual_view ** 2).sum(-1)

    rms_factor = 1.0 / np.sqrt(sqrsum / rms_group_size + mhc_norm_eps)
    mixes = (mixes * rms_factor[:, :, np.newaxis]).sum(-2)

    return mixes.reshape(1, 13, mhc_mult).astype(np.float32)


def compute_golden(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps):
    """计算 norm_fn 的参考输出 (直调通路，residual 为 bf16 uint16 格式).

    Args:
        residual: numpy array, shape (1, 13, 4, 1280), dtype uint16 (bfloat16 编码)
        mhc_fn: numpy array, shape (24, 5120), dtype float32
        mhc_norm_weight: numpy array or None, shape (5120,), dtype float32
        mhc_norm_eps: float

    Returns:
        numpy array, shape (1, 13, 24), dtype float32
    """
    # bf16 (uint16) → float32: 左移 16 位还原
    residual_f32 = residual.view(np.uint16).astype(np.uint32) << 16
    residual_f32 = residual_f32.view(np.float32)
    residual_f32 = residual_f32.reshape(1, 13, 4, 1280)

    residual_f32 = residual_f32.astype(np.float32)
    mhc_fn_f32 = mhc_fn.astype(np.float32)

    if mhc_norm_weight is not None:
        mhc_fn_f32 = mhc_fn_f32 * mhc_norm_weight.astype(np.float32)

    residual_flat = residual_f32.reshape(1, 13, -1)
    mhc_mult = mhc_fn_f32.shape[0]
    rms_group_size = mhc_fn_f32.shape[-1]

    residual_view = residual_flat.reshape(-1, 1, rms_group_size)
    mhc_fn_view = mhc_fn_f32.reshape(mhc_mult, 1, rms_group_size)

    mixes = np.einsum('mbk,nbk->mbn', residual_view, mhc_fn_view)
    sqrsum = (residual_view ** 2).sum(-1)

    rms_factor = 1.0 / np.sqrt(sqrsum / rms_group_size + mhc_norm_eps)
    mixes = (mixes * rms_factor[:, :, np.newaxis]).sum(-2)

    return mixes.reshape(1, 13, mhc_mult).astype(np.float32)
