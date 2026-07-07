/**
 * @file head_compute_mix_fwd_tiling.h
 * @brief Tiling data structure for head_compute_mix_fwd operator.
 *
 * This file is shared by kernel (ASC code) and host (C++ code).
 * Contains only pure C/C++ syntax - no __aicore__, __gm__ etc.
 */

#pragma once

#include <cstdint>

// UB buffer planning constants
// UB total = 192 KB = 196608 bytes (DAV_2201)
// Buffer layout (per element in the tile):
//   inQueue (x2, half):   2 * 2B = 4B
//   f32WorkBuf (x2, float): 2 * 4B = 8B
//   baseF32Expanded (float): 1 * 4B = 4B
//   outQueue (x2, half):  2 * 2B = 4B
// Total per element: 20 bytes
// ubFormer = (196608 / 20) / 128 * 128 = 9728
constexpr int64_t UB_FORMER_HALF = 9728;
constexpr int64_t UB_ALIGN_256B = 256;     // UB 256B alignment
constexpr int64_t ALIGN_256_ELEM_HALF = 128; // 256B / 2B(Half) = 128 elems
constexpr int64_t ALIGN_4_CHECK = 4;       // mhc_mult=4, ubFormer must be divisible by 4

// Multi-core tiling constants (Elementwise standard formula)
constexpr int64_t MIN_TILING_BITS = 32768;  // 4KB minimum per core, in bits
constexpr int64_t ELEM_ALIGN_FACTOR = 512;  // Multi-core element alignment factor
constexpr int32_t MAX_CORE_NUM = 48;        // Max AI Core count on Ascend910B2
constexpr int32_t DOUBLE_BUFFER = 2;        // Double buffer count

/**
 * @brief Tiling data passed from host to device.
 *
 * Contains all runtime parameters needed by the kernel:
 *  - Tile partitioning info
 *  - Scalar parameters (mhc_scale, mhc_pre_eps)
 *  - mhc_base vector (4 elements)
 */
struct HeadComputeMixFwdTilingData {
    // ---- Tile partitioning ----
    int64_t dim0;                     // Total number of elements (batch * n1 * mhc_mult)
    int32_t coreNum;                  // Actual number of cores used
    int64_t blockFormer;              // Elements per core (base), 512-aligned
    int64_t blockNum;                 // Number of blocks (= coreNum)
    int64_t blockTail;                // Elements for the last block (may differ from blockFormer)
    int64_t ubFormer;                 // UB tile size (256B aligned, divisible by 4)
    int64_t ubLoopOfFormerBlock;      // Number of UB tiles in a former block
    int64_t ubTailOfFormerBlock;      // Tail elements in the last UB tile of a former block
    int64_t ubLoopOfTailBlock;        // Number of UB tiles in the tail block
    int64_t ubTailOfTailBlock;        // Tail elements in the last UB tile of the tail block

    // ---- Scalar parameters (FP32 for precision) ----
    float  mhc_scale_f32;
    float  mhc_pre_eps_f32;

    // ---- mhc_base vector (FP32, 4 elements) ----
    // Stored as FP32 to avoid half type in pure C/C++ header
    float mhc_base_f32[4];
};
