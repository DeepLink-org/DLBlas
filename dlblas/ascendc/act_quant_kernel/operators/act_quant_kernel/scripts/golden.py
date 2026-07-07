# ----------------------------------------------------------------------------
# Golden 计算（双通路共用）
#
# act_quant_kernel: per-group FP8 quantization
# 输入: x (numpy/torch, bf16/fp16), group_size, eps, scale_ue8m0
# 输出: x_q (fp8_e4m3fn as uint8), x_s (fp32)
# ----------------------------------------------------------------------------

import numpy as np


def fp32_to_fp8_e4m3fn(val):
    """Convert a single fp32 value to fp8_e4m3fn uint8 encoding.

    DAV_2201 software implementation mirroring the kernel's scalar conversion.
    """
    import struct

    if np.isnan(val):
        return 0x7F

    # Extract fp32 bits
    bits = np.float32(val).view(np.uint32)
    sign = (bits >> 31) & 1
    exp  = int((bits >> 23) & 0xFF) - 127
    mant = int(bits & 0x7FFFFF)

    # Zero
    if exp == -127 and mant == 0:
        return int(sign << 7)

    # Overflow -> max (448 = 0x7E)
    if exp > 8:
        return int((sign << 7) | 0x7E)

    e8 = exp + 7

    # Subnormal
    if e8 <= 0:
        if exp < -9:
            return int(sign << 7)
        mfull = mant | 0x800000
        shift = 20 + (-e8 + 1)
        if shift >= 24:
            return int(sign << 7)
        m8 = mfull >> shift
        # Round
        if shift > 0:
            round_bit = (mfull >> (shift - 1)) & 1
            if round_bit:
                sticky = (mfull & ((1 << (shift - 1)) - 1)) != 0 if shift >= 2 else False
                if sticky or (m8 & 1):
                    m8 += 1
        if m8 > 7:
            m8 = 7
        return int((sign << 7) | (m8 & 0x7))

    # Normal
    if e8 > 15:
        e8 = 15

    m8 = mant >> 20
    round_bit = (mant >> 19) & 1
    sticky = (mant & 0x7FFFF) != 0

    if round_bit and (sticky or (m8 & 1)):
        m8 += 1
        if m8 == 8:
            m8 = 0
            e8 += 1
            if e8 > 15:
                e8 = 15

    return int((sign << 7) | ((e8 & 0xF) << 3) | (m8 & 0x7))


# Vectorized version for numpy arrays
fp32_to_fp8_vec = np.vectorize(fp32_to_fp8_e4m3fn, otypes=[np.uint8])


def fp8_e4m3fn_to_fp32(uint8_val):
    """Decode fp8_e4m3fn uint8 -> fp32 value. For validation purposes."""
    sign = (uint8_val >> 7) & 1
    exp  = (uint8_val >> 3) & 0xF
    mant = uint8_val & 0x7

    if exp == 0:
        # Subnormal or zero
        val = np.float32((1 - 2 * sign) * (mant / 8.0) * (2.0 ** (-6)))
    elif exp == 15:
        # Normal (no Inf in fp8_e4m3fn; m=7 is NaN)
        if mant == 7:
            return np.float32(np.nan)
        val = np.float32((1 - 2 * sign) * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))
    else:
        val = np.float32((1 - 2 * sign) * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))

    return val


# Vectorized decode
fp8_to_fp32_vec = np.vectorize(fp8_e4m3fn_to_fp32, otypes=[np.float32])


def compute_golden(x, group_size=128, eps=1e-10, scale_ue8m0=False):
    """Compute act_quant_kernel golden output.

    Args:
        x: numpy array (1D flattened, bf16 or fp16 → upcast to float32)
        group_size: elements per group
        eps: amax lower clamp
        scale_ue8m0: whether to use UE8M0 scale format (not implemented yet)

    Returns:
        (x_q_uint8, x_s_fp32): tuple of golden outputs
    """
    x_f32 = x.astype(np.float32).reshape(-1, group_size)
    num_groups = x_f32.shape[0]

    x_s = np.zeros(num_groups, dtype=np.float32)
    x_q = np.zeros(x_f32.size, dtype=np.uint8)

    fp8_max = np.float32(448.0)

    for g in range(num_groups):
        row = x_f32[g]
        amax = np.max(np.abs(row))
        if amax < eps or np.isnan(amax):
            amax = np.float32(eps)
        scale = amax / fp8_max

        if scale_ue8m0:
            # UE8M0: exp2(ceil(log2(max(|scale|, 1e-10))))
            abs_s = abs(scale)
            if abs_s < 1e-10:
                abs_s = 1e-10
            log2_s = np.log2(abs_s)
            ceil_log2 = np.ceil(log2_s)
            scale = np.float32(2.0 ** ceil_log2)

        x_s[g] = scale
        quantized = row / scale
        quantized = np.clip(quantized, -448.0, 448.0)
        x_q[g * group_size:(g + 1) * group_size] = fp32_to_fp8_vec(quantized)

    return x_q, x_s
