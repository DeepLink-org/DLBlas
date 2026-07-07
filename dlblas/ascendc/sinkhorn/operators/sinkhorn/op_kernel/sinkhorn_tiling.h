/**
 * Sinkhorn Normalize - Tiling 常量和结构体
 * kernel 和 host 共用，只含纯 C/C++ 语法
 */

#pragma once

#include <cstdint>

// 矩阵维度固定为 4
constexpr uint32_t MHC = 4;
constexpr uint32_t MATRIX_SIZE = MHC * MHC;  // 16 floats per matrix
constexpr float EPS = 1e-6f;
constexpr uint32_t REPEAT = 10;

// DataCopyPad blockCount 最大值为 4095
// tile_batch 需要满足 tile_batch * MATRIX_SIZE <= 4095
// MATRIX_SIZE=16，所以 tile_batch <= 255
constexpr uint32_t MAX_TILE_MATRICES = 255;

// Tiling 数据结构 - 向核函数传递的运行时参数
struct SinkhornTilingData {
    uint32_t batch;          // 总 batch 数
    uint32_t mhc;            // 矩阵维度 = 4
    uint32_t repeat;         // 迭代次数 = 10
    float eps;               // epsilon = 1e-6
    uint32_t tileBatch;      // 每核处理的矩阵数
    uint32_t tailBatch;      // 尾核矩阵数
    uint32_t usedCoreNum;    // 实际使用的核数
    uint32_t tileElements;   // 本核处理的元素数 (tileBatch or tailBatch * 16)
};
