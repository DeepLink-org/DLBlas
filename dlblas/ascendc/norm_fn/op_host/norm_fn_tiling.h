/**
 * norm_fn Tiling 参数定义
 *
 * 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字。
 * kernel (.asc) 和 host (.asc / .cpp) 共用此文件。
 */

#pragma once

#include <cstdint>

// ============================================================================
// 编译期常量
// ============================================================================

// 问题规模
constexpr uint32_t TOTAL_M = 13;     // 输出行数 (residual 展平后的 M)
constexpr uint32_t TOTAL_N = 24;     // 输出列数 (mhc_fn 行数)
constexpr uint32_t TOTAL_K = 5120;   // 内维大小

// K 轴分块
constexpr uint32_t TILE_K       = 512;   // K 轴 tile 大小
constexpr uint32_t TILE_K_ALIGN = 512;   // 32B 对齐后 (512*4=2048 已对齐)
constexpr uint32_t NUM_K_TILES  = 10;    // TOTAL_K / TILE_K = 5120/512

// UB Buffer 大小参数
constexpr uint32_t RESIDUAL_ROWS      = 13;    // M (13)
constexpr uint32_t MHC_FN_ROWS        = 24;    // N (24)

// 各 Buffer 元素数
constexpr uint32_t RESIDUAL_ELEMS  = RESIDUAL_ROWS  * TILE_K_ALIGN;  // 13*512 = 6656
constexpr uint32_t MHC_FN_ELEMS    = MHC_FN_ROWS    * TILE_K_ALIGN;  // 24*512 = 12288
constexpr uint32_t WEIGHT_ELEMS    = TILE_K_ALIGN;                   // 512
constexpr uint32_t SQ_TEMP_ELEMS   = RESIDUAL_ROWS  * TILE_K_ALIGN;  // 6656 (also tempRow)
constexpr uint32_t MIXES_ELEMS     = TOTAL_M * TOTAL_N;              // 13*24 = 312
constexpr uint32_t SQSUM_ELEMS     = TOTAL_M;                        // 13
constexpr uint32_t RESULT_ELEMS    = TOTAL_M * TOTAL_N;              // 312

// Reduce 临时 Buffer (足够大)
constexpr uint32_t REDUCE_TMP_SIZE = 64 * 1024;  // 64KB

// Double Buffer 策略: 不使用 (K 轴仅 10 次迭代，单 Buffer 即可)
constexpr uint32_t QUE_DEPTH = 1;

// ============================================================================
// Tiling 数据结构 (Host → Device)
// ============================================================================

struct NormFnTilingData {
    // 输入形状
    uint32_t total_M;        // 13
    uint32_t total_N;        // 24
    uint32_t total_K;        // 5120

    // K 轴分块参数
    uint32_t tile_K;         // 512
    uint32_t tile_K_align;   // 512
    uint32_t num_K_tiles;    // 10

    // 是否使用权重
    bool     has_weight;

    // eps 值 (1e-6)
    float    eps;

    // 预计算的 invK = 1.0f / total_K (aicore 中不能 cast uint32→float)
    float    invK;
};
