# apply_mix golden computation (numpy-based reference)
# double-pathway: used by gen_data.py and test_torch.py

import numpy as np
import struct


def fp32_to_bf16_uint16(arr_fp32):
    """Convert float32 array to bfloat16 (return uint16 for binary write)."""
    arr = arr_fp32.astype(np.float32).ravel()
    result = np.zeros(len(arr), dtype=np.uint16)
    for i, v in enumerate(arr):
        u32 = struct.unpack('I', struct.pack('f', float(v)))[0]
        # Round to nearest even
        rounding = ((u32 >> 16) & 1) + 0x7FFF
        u32_rounded = u32 + rounding
        result[i] = np.uint16(u32_rounded >> 16)
    return result


def bf16_uint16_to_fp32(arr_bf16_uint16):
    """Convert bf16 uint16 to float32."""
    arr = arr_bf16_uint16.astype(np.uint32).ravel()
    result = np.zeros(len(arr), dtype=np.float32)
    for i, v in enumerate(arr):
        u32 = np.uint32(v) << 16
        result[i] = struct.unpack('f', struct.pack('I', int(u32)))[0]
    return result


def compute_golden(x_bf16_raw, mix_fp32, n0, n1, mhc, h):
    """Compute the reference output of apply_mix.

    Args:
        x_bf16_raw: numpy array of shape [n0, n1, mhc, h], dtype uint16 (bf16 raw)
        mix_fp32: numpy array of shape [n0, n1, mhc, 1], dtype float32
        n0, n1, mhc, h: shape parameters

    Returns:
        numpy array of shape [n0, n1, h], dtype uint16 (bf16 output)
    """
    # Convert x from bf16 uint16 to float32
    x_fp32 = bf16_uint16_to_fp32(x_bf16_raw).reshape(n0, n1, mhc, h)

    # Reshape mix to [n0, n1, mhc, 1]
    mix = mix_fp32.reshape(n0, n1, mhc, 1).astype(np.float32)

    # Multiply with broadcast: [n0,n1,mhc,h] * [n0,n1,mhc,1] → [n0,n1,mhc,h]
    prod = x_fp32 * mix

    # Sum along mhc axis (axis=2): [n0,n1,mhc,h] → [n0,n1,h]
    result_fp32 = np.sum(prod, axis=2)

    # Convert to bf16
    return fp32_to_bf16_uint16(result_fp32)
