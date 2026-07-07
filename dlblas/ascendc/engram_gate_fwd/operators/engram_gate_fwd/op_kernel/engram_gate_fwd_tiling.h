/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Tiling data structures for engram_gate_fwd operator.
 * This file contains pure C/C++ (no ASC keywords) shared between kernel and host.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <cstddef>

// [CONFIG] UB capacity: DAV_2201 = 192KB = 196608 bytes
constexpr size_t UB_CAPACITY_DAV_2201 = 192 * 1024;
constexpr size_t TILING_RESERVED = 2048;             // Reserve for runtime overhead

// Compute safe maximum hidden_size for given hc_mult
inline size_t MaxHiddenSizeForUB(size_t hc_mult, size_t ub_capacity = UB_CAPACITY_DAV_2201) {
    // UB budget estimate: weight_row * 2 + bf16_row * 5 + fp32_row * 3 + misc
    // bf16_row = ceil(hs*2/32)*32, fp32_row = ceil(hs*4/32)*32/4*4
    // For hs aligned to 8 (fp32 32B): bf16_row = hs*2, fp32_row = hs*4
    // UB = hs*2*7 + hs*4*3 + misc = hs*26 + misc
    // hs_max = (ub - misc - weight_extra) / 26
    constexpr size_t MISC_BYTES = 4096;
    size_t weight_extra = (hc_mult > 1) ? (hc_mult - 1) * 0 : 0; // single-row weight loading
    // Conservative: each bf16 row = ceil(hs*2/32)*32, each fp32 row = ceil(hs*4/32)*32
    // For hs aligned: 6 bf16 rows + 3 fp32 rows = 6*hs*2 + 3*hs*4 = hs*24
    // Plus weight (2 rows): 2*hs*2 = hs*4. Total = hs*28 + misc
    size_t available = ub_capacity - TILING_RESERVED - MISC_BYTES;
    // For hidden_size aligned to 8, each element costs 28 bytes (6*2 + 3*4 + 2*2)
    // But with padding: bf16 row = ceil(hs*2/32)*32, fp32 row = ceil(hs*4/32)*32
    // Approximate: hs_max = available / 28
    return available / 28;
}

struct EngramGateFwdTilingData {
    // Dimensions
    uint64_t num_tokens;
    uint64_t hc_mult;
    uint64_t hidden_size;
    uint64_t hidden_size_align;        // 32B-aligned element count for fp32
    uint64_t hidden_size_align_bf16;   // 32B-aligned element count for bf16

    // Multi-core split
    uint64_t tile_rows_per_core;  // rows per core (aligned to hc_mult)
    uint64_t total_rows;          // total rows = num_tokens * hc_mult
    uint32_t core_num;            // number of cores used

    // Scalar parameters
    float clamp_value;
    float eps;
    float scalar;                  // hidden_size^(-0.5)
    float hidden_size_float;      // hidden_size as float (for aicore division)

    // Pointer offsets (GM base address offsets, in bytes)
    uint64_t hidden_states_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t weight_hidden_offset;
    uint64_t weight_embed_offset;
    uint64_t output_offset;
    uint64_t raw_dot_offset;
    uint64_t gate_score_offset;
    uint64_t rstd_x_offset;
    uint64_t rstd_k_offset;
};

// Compute aligned byte count for 32B alignment
inline constexpr uint64_t AlignTo32B(uint64_t bytes) {
    return ((bytes + 31) / 32) * 32;
}

// Compute total UB buffer bytes for given hidden_size and hc_mult
// Matches actual kernel buffer layout with lazy weight loading (single row each):
//   - weight_hidden_q_: hbb_align (lazy, one row per head)
//   - weight_embed_q_:  hbb_align (lazy, one row per head)
//   - v_q_, x_q_, k_q_, out_q_: each hbb_align (4 row buffers)
//   - buf_a_q_, buf_b_q_, buf_c_q_: each hfb_align (3 fp32 work buffers)
//   - tmp_q_: 8192 (Reduce tmpBuf)
//   - scalar_q_: 32 (scalar write buffer)
inline uint64_t ComputeUBUsage(uint64_t hidden_size, uint64_t hc_mult) {
    (void)hc_mult;  // not used for lazy loading
    uint64_t hbb_align = AlignTo32B(hidden_size * sizeof(uint16_t));   // bf16 aligned bytes per row
    uint64_t hfb_align = AlignTo32B(hidden_size * sizeof(float));      // fp32 aligned bytes per row

    uint64_t bf16_buf_bytes = hbb_align * 6;   // weight_hidden, weight_embed, v, x, k, out
    uint64_t fp32_buf_bytes = hfb_align * 3;   // buf_a, buf_b, buf_c
    uint64_t tmp_buf_bytes = 8192;
    uint64_t scalar_bytes = 32;

    return bf16_buf_bytes + fp32_buf_bytes + tmp_buf_bytes + scalar_bytes;
}

// Compute tiling parameters on host side
inline void ComputeTiling(EngramGateFwdTilingData& tiling,
                          uint64_t num_tokens, uint64_t hc_mult, uint64_t hidden_size,
                          float clamp_value, float eps, uint32_t core_num)
{
    tiling.num_tokens = num_tokens;
    tiling.hc_mult = hc_mult;
    tiling.hidden_size = hidden_size;

    // hidden_size_align: round up to 32B-aligned fp32 elements (8 floats per 32B)
    tiling.hidden_size_align =
        ((hidden_size * sizeof(float) + 31) / 32) * 32 / sizeof(float);

    // hidden_size_align_bf16: round up to 32B-aligned bf16 elements
    tiling.hidden_size_align_bf16 =
        ((hidden_size * sizeof(uint16_t) + 31) / 32) * 32 / sizeof(uint16_t);

    tiling.total_rows = num_tokens * hc_mult;
    tiling.clamp_value = clamp_value;
    tiling.eps = eps;
    tiling.scalar = 1.0f / sqrtf(static_cast<float>(hidden_size));
    tiling.hidden_size_float = static_cast<float>(hidden_size);

    // Multi-core split
    tiling.core_num = core_num;
    tiling.tile_rows_per_core =
        (tiling.total_rows + core_num - 1) / core_num;
    // Align to hc_mult to maintain token integrity
    tiling.tile_rows_per_core =
        ((tiling.tile_rows_per_core + hc_mult - 1) / hc_mult) * hc_mult;

    // Offsets (host computes per-core GM offsets for each tensor base)
    tiling.hidden_states_offset = 0;
    tiling.k_offset = 0;
    tiling.v_offset = 0;
    tiling.weight_hidden_offset = 0;
    tiling.weight_embed_offset = 0;
    tiling.output_offset = 0;
    tiling.raw_dot_offset = 0;
    tiling.gate_score_offset = 0;
    tiling.rstd_x_offset = 0;
    tiling.rstd_k_offset = 0;
}
