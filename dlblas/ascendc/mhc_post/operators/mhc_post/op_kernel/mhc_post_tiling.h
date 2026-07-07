/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

/* MHC Post 算子 Tiling 结构体和常量 - kernel 和 host 共用 */
/* 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字 */

#pragma once

#include <cstdint>

// ============================================================================
// Tile 尺寸常量
// ============================================================================
// B_TILE: batch tile 大小（DESIGN v2: 逐 batch 处理 = 1）
//         系数数据仅 80B/batch，逐 batch 加载开销可忽略（占总搬运 0.34%）
constexpr uint32_t B_TILE = 1;
// C_TILE: column tile 大小，32 字节对齐: 64 * 2 = 128B = 4 * 32B
constexpr uint32_t C_TILE = 64;

// ============================================================================
// 固定维度常量（默认值；可通过命令行参数覆盖）
// ============================================================================
constexpr uint32_t N0_DEFAULT = 2;           // dim 0 大小（默认）
constexpr uint32_t N1_DEFAULT = 4096;        // dim 1 大小（默认）
constexpr uint32_t H_DEFAULT  = 1280;        // dim 2 大小（h 维度，默认）
constexpr uint32_t MHC_MULT = 4;             // 多头压缩倍数（固定）
constexpr uint32_t MAX_CORE_NUM = 20;        // 最大使用核数

// ============================================================================
// Tiling 数据结构 - 向核函数传递的运行时参数
// ============================================================================
struct MhcPostTiling {
    uint32_t n0;              // dim 0 大小（如 2）
    uint32_t blockNum;        // 使用的核数（用于核内计算 n1 范围）
    uint32_t bTile;           // = 1, batch tile（逐 batch 处理）
    uint32_t cTile;           // = 64, column tile
    uint32_t h;               // h 维度大小（如 1280）
    uint32_t n1;              // dim 1 大小（如 4096）
};
