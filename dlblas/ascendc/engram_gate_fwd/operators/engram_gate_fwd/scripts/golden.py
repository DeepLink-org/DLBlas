# --------------------------------------------------------------------------
# Golden computation for engram_gate_fwd
# --------------------------------------------------------------------------
# Reference implementation in fp32. Used by gen_data.py and test_torch.py.
# --------------------------------------------------------------------------

import numpy as np


def fp32_to_bf16(x):
    """Convert fp32 to bf16 (stored as uint16) with round-to-nearest-even.

    bf16 = upper 16 bits of fp32, with rounding based on lower 16 bits.
    Matches AscendC Cast with RoundMode::CAST_ROUND behavior.
    """
    x_f32 = np.asarray(x, dtype=np.float32)
    x_u32 = x_f32.view(np.uint32)
    # Extract upper 16 bits (potential bf16 value)
    upper = (x_u32 >> 16).astype(np.uint32)
    # Extract lower 16 bits for rounding decision
    lower = x_u32 & 0xFFFF
    # Round to nearest even:
    # - If lower > 0x8000: round up
    # - If lower == 0x8000 and upper is odd: round up (nearest even)
    # - Otherwise: keep upper (round down)
    round_up = (lower > 0x8000) | ((lower == 0x8000) & ((upper & 1) != 0))
    # Handle NaN/Inf: don't round them
    is_special = (upper & 0x7F80) == 0x7F80  # exponent all 1s = Inf/NaN
    round_up = round_up & ~is_special
    upper = upper + round_up.astype(np.uint32)
    return upper.astype(np.uint16)


def bf16_to_fp32(x_bf16):
    """Convert bf16 (as uint16) to fp32."""
    x_u32 = np.asarray(x_bf16, dtype=np.uint32) << 16
    x_f32 = x_u32.view(np.float32)
    return np.asarray(x_f32, dtype=np.float32)


def compute_golden(hs_f32, k_f32, v_f32, wh_f32, we_f32, clamp_value, eps, hidden_size):
    """Compute reference output for engram_gate_fwd in fp32.

    Args:
        hs_f32:  [num_tokens, hc_mult, hidden_size] fp32
        k_f32:   [num_tokens, hc_mult, hidden_size] fp32
        v_f32:   [num_tokens, hidden_size] fp32
        wh_f32:  [hc_mult, hidden_size] fp32
        we_f32:  [hc_mult, hidden_size] fp32
        clamp_value: float
        eps: float
        hidden_size: int

    Returns:
        output_f32:  [num_tokens, hc_mult, hidden_size] fp32
        raw_dot:     [num_tokens, hc_mult] fp32
        gate_score:  [num_tokens, hc_mult] fp32
        rstd_x:      [num_tokens, hc_mult] fp32
        rstd_k:      [num_tokens, hc_mult] fp32
    """
    num_tokens, hc_mult, hidden_size = hs_f32.shape
    scalar = hidden_size ** -0.5

    # RMSNorm: rstd = 1/sqrt(mean(x^2) + eps)
    rstd_x = 1.0 / np.sqrt(np.mean(hs_f32 ** 2, axis=-1) + eps)
    rstd_k = 1.0 / np.sqrt(np.mean(k_f32 ** 2, axis=-1) + eps)

    # Dot product: raw_dot = sum(x * wh * k * we)
    wh_expanded = wh_f32[np.newaxis, :, :]      # [1, hc_mult, hidden_size]
    we_expanded = we_f32[np.newaxis, :, :]      # [1, hc_mult, hidden_size]
    x_w = hs_f32 * wh_expanded                   # [num_tokens, hc_mult, hidden_size]
    k_w = k_f32 * we_expanded                    # [num_tokens, hc_mult, hidden_size]
    raw_dot = np.sum(x_w * k_w, axis=-1)         # [num_tokens, hc_mult]

    # dot = raw_dot * rstd_x * rstd_k * scalar
    dot = raw_dot * rstd_x * rstd_k * scalar

    # signed_sqrt = sign(dot) * sqrt(max(|dot|, clamp_value))
    abs_dot = np.abs(dot)
    clamped = np.maximum(abs_dot, clamp_value)
    signed_sqrt = np.copysign(np.sqrt(clamped), dot)

    # gate_score = sigmoid(signed_sqrt)
    gate_score = 1.0 / (1.0 + np.exp(-signed_sqrt))

    # output = x + gate_score * v (broadcast gate_score)
    output_f32 = hs_f32 + gate_score[:, :, np.newaxis] * v_f32[:, np.newaxis, :]

    return output_f32, raw_dot, gate_score, rstd_x, rstd_k
