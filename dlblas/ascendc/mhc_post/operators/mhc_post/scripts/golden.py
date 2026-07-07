# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Golden 计算 - MHC Post（直调 & PyTorch 双通路共用）
#
# term2[a,b,:,:] = comb_res_mix[a,b,:,:] @ residual[a,b,:,:]
# output[a,b,:,:] = bf16(x[a,b,:] * post_layer_mix[a,b,:,0] + term2[a,b,:,:])
#
# bf16 precision: kernel uses CAST_NONE (no rounding, zero-extend) for bf16→fp32
#                 and CAST_ROUND (round-to-nearest-even) for fp32→bf16
# ============================================================================

import numpy as np
import sys
import os


# ============================================================================
# bf16 utility functions (numpy vectorized)
# ============================================================================

def fp32_to_bf16(arr_fp32: np.ndarray) -> np.ndarray:
    """
    Convert fp32 numpy array to bf16 (stored as uint16).
    Uses simple truncation (right shift by 16), matching what happens
    when bf16 data is stored in memory.
    """
    orig_shape = arr_fp32.shape
    arr_flat = arr_fp32.ravel().view(np.int32)
    bf16_flat = (arr_flat >> 16).astype(np.uint16)
    return bf16_flat.reshape(orig_shape)


def fp32_to_bf16_rne(arr_fp32: np.ndarray) -> np.ndarray:
    """
    Convert fp32 to bf16 with round-to-nearest-even (matches CAST_ROUND).
    RNE: round up if low 16 bits > 0x8000, or if == 0x8000 and bit 16 is 1.
    """
    orig_shape = arr_fp32.shape
    arr_flat = arr_fp32.ravel().view(np.int32)
    low16 = arr_flat & 0xFFFF
    high16 = arr_flat >> 16

    # Round up when low16 > 0x8000, or low16 == 0x8000 and LSB of high16 is 1
    round_up = ((low16 > 0x8000) | ((low16 == 0x8000) & (high16 & 1))).astype(np.int32)

    result = (high16 + round_up).astype(np.uint16)
    return result.reshape(orig_shape)


def bf16_to_fp32(arr_bf16: np.ndarray) -> np.ndarray:
    """
    Convert bf16 (stored as uint16) back to fp32 by zero-extending
    (matches CAST_NONE: left shift by 16, pad with zeros).
    """
    orig_shape = arr_bf16.shape
    arr_flat = arr_bf16.ravel().astype(np.uint32)
    fp32_flat = (arr_flat << 16).view(np.float32)
    return fp32_flat.reshape(orig_shape)


def write_bf16_bin(arr_fp32: np.ndarray, filepath: str):
    """Write fp32 array as bf16 binary file (simple truncation)."""
    bf16_data = fp32_to_bf16(arr_fp32)
    bf16_data.tofile(filepath)


def read_bf16_bin(filepath: str, shape: tuple) -> np.ndarray:
    """Read bf16 binary file, convert to fp32 for computation."""
    data = np.fromfile(filepath, dtype=np.uint16)
    return bf16_to_fp32(data).reshape(shape)


def make_data(n0: int, n1: int, h: int, mhc_mult: int, seed: int = None):
    """Generate test data as numpy arrays (fp32)."""
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randn(n0, n1, h).astype(np.float32)
    residual = np.random.randn(n0, n1, mhc_mult, h).astype(np.float32)
    post_layer_mix = np.random.randn(n0, n1, mhc_mult, 1).astype(np.float32)
    comb_res_mix = np.random.randn(n0, n1, mhc_mult, mhc_mult).astype(np.float32)
    return x, residual, post_layer_mix, comb_res_mix


def compute_golden(x, residual, post_layer_mix, comb_res_mix):
    """
    Compute reference output for MHC Post at bf16 precision.

    Simulates the kernel precision path:
      1. bf16 inputs → fp32 (CAST_NONE = zero-extend)
      2. fp32 computation
      3. fp32 → bf16 output (CAST_ROUND = RNE)

    Args:
        x: (n0, n1, h) fp32
        residual: (n0, n1, 4, h) fp32
        post_layer_mix: (n0, n1, 4, 1) fp32
        comb_res_mix: (n0, n1, 4, 4) fp32

    Returns:
        output: (n0, n1, 4, h) fp32 (computed at bf16 precision, then extended back to fp32)
    """
    n0, n1, h = x.shape
    mhc_mult = comb_res_mix.shape[-1]

    # Step 1: simulate bf16→fp32 for inputs (CAST_NONE = zero-extend)
    # This is equivalent to: convert to bf16 (truncate) then back to fp32 (zero-extend)
    x_bf16_to_fp32 = bf16_to_fp32(fp32_to_bf16(x))
    residual_bf16_to_fp32 = bf16_to_fp32(fp32_to_bf16(residual))

    # Step 2: compute in fp32 (matching reference mhc_post.py which uses .float())
    # Use sequential accumulation (matching kernel's Muls+Add order) for precise match.
    x_fp32_arr = x_bf16_to_fp32.astype(np.float32)
    residual_fp32_arr = residual_bf16_to_fp32.astype(np.float32)
    post_layer_mix_fp32 = post_layer_mix.astype(np.float32)
    comb_res_mix_fp32 = comb_res_mix.astype(np.float32)

    # term2[a,b,m,c] = sum_{k=0}^{3} cmb[a,b,m,k] * res[a,b,k,c]
    # Use sequential accumulation matching kernel's Muls+Add order:
    #   term2 = cmb[m,0]*res[0] + cmb[m,1]*res[1] + cmb[m,2]*res[2] + cmb[m,3]*res[3]
    term2 = comb_res_mix_fp32[:, :, :, 0:1] * residual_fp32_arr[:, :, 0:1, :]  # init
    for k in range(1, mhc_mult):
        tmp = comb_res_mix_fp32[:, :, :, k:k+1] * residual_fp32_arr[:, :, k:k+1, :]
        term2 = term2 + tmp  # fp32 accumulation

    # output = x * post_layer_mix + term2 (broadcast x along mhc_mult axis)
    output_fp32 = (x_fp32_arr[:, :, np.newaxis, :] * post_layer_mix_fp32 + term2).astype(np.float32)

    # Step 3: simulate fp32→bf16 output (CAST_ROUND = RNE), then extend back to fp32
    output_bf16 = fp32_to_bf16_rne(output_fp32)

    return bf16_to_fp32(output_bf16).astype(np.float32)


# ============================================================================
# Test case definitions (matched to PLAN.md §2.1-2.2: TC-01 ~ TC-09)
# ============================================================================
# Format: (tag, n0, n1, h, mhc_mult, seed, description)
TEST_CASES = [
    ("TC-01", 2, 4096, 1280, 4, 42, "Standard shape"),
    ("TC-02", 1, 1,    64,   4, 43, "Minimal shape (single batch, single col tile)"),
    ("TC-03", 1, 1,    1,    4, 44, "Smallest shape (C_TILE > h extreme)"),
    ("TC-04", 1, 16,   1280, 4, 45, "Small n1 (few cores)"),
    ("TC-05", 2, 4096, 64,   4, 46, "Minimal h (= C_TILE, single column tile)"),
    ("TC-06", 2, 4097, 1280, 4, 47, "n1 not divisible by typical core count (tail logic)"),
    ("TC-07", 1, 1,    1280, 4, 48, "n1 < blockNum (idle core skip)"),
    ("TC-08", 2, 4096, 1280, 4, 49, "All-zero input"),
    ("TC-09", 2, 4096, 1280, 4, 50, "Extreme value input"),
    ("TC-10", 2, 4096, 1280, 4, 51, "Mixed sign input"),
    ("TC-11", 1, 4096, 1280, 4, 52, "Single batch axis (n0=1)"),
    ("TC-12", 2, 4096, 130,  4, 53, "h not divisible by C_TILE (tail block)"),
]


def get_test_cases():
    """Return list of test case tuples."""
    return TEST_CASES


def generate_test_data(tag, n0, n1, h, mhc_mult, seed):
    """Generate test data for a specific test case."""
    if tag == "TC-08":
        # All-zero input
        x = np.zeros((n0, n1, h), dtype=np.float32)
        residual = np.zeros((n0, n1, mhc_mult, h), dtype=np.float32)
        post_layer_mix = np.zeros((n0, n1, mhc_mult, 1), dtype=np.float32)
        comb_res_mix = np.zeros((n0, n1, mhc_mult, mhc_mult), dtype=np.float32)
    elif tag == "TC-09":
        # Extreme value input (near bf16 max ~3.39e38)
        np.random.seed(seed)
        x = (np.random.randn(n0, n1, h) * 10000).astype(np.float32)
        residual = (np.random.randn(n0, n1, mhc_mult, h) * 10000).astype(np.float32)
        post_layer_mix = (np.random.randn(n0, n1, mhc_mult, 1) * 10).astype(np.float32)
        comb_res_mix = (np.random.randn(n0, n1, mhc_mult, mhc_mult) * 10).astype(np.float32)
    elif tag == "TC-10":
        # Mixed sign input
        np.random.seed(seed)
        x = (np.random.randn(n0, n1, h) * 5).astype(np.float32)
        residual = (np.random.randn(n0, n1, mhc_mult, h) * 5).astype(np.float32)
        post_layer_mix = (np.random.randn(n0, n1, mhc_mult, 1) * 2).astype(np.float32)
        comb_res_mix = (np.random.randn(n0, n1, mhc_mult, mhc_mult) * 2).astype(np.float32)
    else:
        return make_data(n0, n1, h, mhc_mult, seed)
    return x, residual, post_layer_mix, comb_res_mix
