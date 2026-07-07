/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#pragma once

#include <cstdint>

// 最小对齐单位: half 类型, VECTOR_REG_WIDTH=256 bits / 16 bits = 128
constexpr uint32_t A0_TILE_BASE = 128;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct ExpandKernelBwdTilingData {
    uint64_t A1;            // 外层保留轴总大小 = n0 * n1
    uint64_t R;             // 归约轴大小 = mhc_mult
    uint64_t A0;            // 内层保留轴总大小 = h
    uint64_t tileA0Len;     // UB 切片 A0 大小 (对齐到 A0_TILE_BASE)
    uint64_t a0Outer;       // A0 切片份数
    uint64_t totalTiles;    // 总 tile 数 = A1 * a0Outer
    uint64_t tilesPerCore;  // 每核 tile 数
    uint64_t tailCoreTiles; // 尾核 tile 数
    uint64_t usedCoreNum;   // 使用核数
    uint32_t inputSize;     // 输入总大小 (bytes)
    uint32_t outputSize;    // 输出总大小 (bytes)
};
