/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/* hc_split_sinkhorn Tiling - kernel 和 host 共用 */

#pragma once

#include <cstdint>

// ============================================================================
// 编译期常量
// ============================================================================

#define HC_SPLIT_SINKHORN_MAX_HC 32
#define HC_SPLIT_SINKHORN_MAX_MIX_HC ((2 + HC_SPLIT_SINKHORN_MAX_HC) * HC_SPLIT_SINKHORN_MAX_HC)

constexpr uint64_t UB_SIZE = 192 * 1024;         // DAV_2201 UB 容量
constexpr uint64_t REDUCE_TMP_BUF_SIZE = 4096;   // Reduce API 最小空间 (bytes)
constexpr uint64_t BASE_START = 8;               // param 布局: 8-float 边界对齐
constexpr uint64_t COMPUTE_BUF_SIZE = 128;       // 常量缓冲区 (ones, neg, two)

// ============================================================================
// Tiling 数据结构
// ============================================================================

struct HcSplitSinkhornTiling {
    // 基础 shape 信息
    uint64_t totalBatch;         // B = b * s
    uint64_t hc;                 // hc 维度
    uint64_t mixHc;              // mix_hc = (2+hc)*hc
    uint32_t sinkhornIters;      // Sinkhorn 迭代次数 (>= 1)
    float    eps;                // 数值稳定常数

    // 对齐信息
    uint64_t mixHcAlign;         // 32B 对齐后的 mix_hc（每行元素数）
    uint64_t hcAlign;            // 32B 对齐后的 hc

    // 多核切分
    uint64_t rowsPerCore;        // 每核处理的行数
    uint32_t tailCoreRows;       // 尾核行数
    uint32_t usedCoreNum;        // 实际使用的核数

    // UB 切分
    uint32_t tileRows;           // 每 tile 行数 (T)
    uint32_t tilesPerCore;       // 每核 tile 数

    // 参数（固定大小数组，编译期 MAX_HC 约束）
    float hcScale[3];
    float hcBase[HC_SPLIT_SINKHORN_MAX_MIX_HC];
};

// ============================================================================
// 工具函数
// ============================================================================

// 将字节数向上对齐到 32
constexpr uint64_t align32_up(uint64_t n) {
    return ((n + 31) / 32) * 32;
}

inline uint64_t align32_elements(uint64_t nElements) {
    return align32_up(nElements * sizeof(float)) / sizeof(float);
}

// ============================================================================
// UB Tile 行数计算
// ============================================================================

// 计算单 tile 可处理的最大行数，基于 UB 容量约束
// 所有 buffer 在 Init 时一次性分配，取全部 buffer 的总和为约束
inline uint64_t calcTileRows(uint64_t hc, uint64_t mixHcAlign, uint64_t hcAlign) {
    uint64_t hcHc = hc * hc;

    // 参数缓冲区大小（包含 32B 对齐后的 total）
    uint64_t postBaseOff = ((BASE_START + hc + 7) / 8) * 8;
    uint64_t combBaseOff = ((postBaseOff + hc + 7) / 8) * 8;
    uint64_t paramFloats = combBaseOff + hcHc;
    uint64_t paramBufSize = align32_up(paramFloats * sizeof(float));

    // 固定开销（不随 T 变化）
    uint64_t fixedBytes = paramBufSize + REDUCE_TMP_BUF_SIZE + COMPUTE_BUF_SIZE;
    if (fixedBytes >= UB_SIZE) return 1;

    // 全部并发 buffer 总开销: inQueue + 2*outQueue + 4*workBuf + tmpBuf
    // = mixHcAlign + hcAlign + hcAlign + hc*hcAlign + hc + hc + hc*hc + hcAlign
    uint64_t perRowFloats = mixHcAlign + 3 * hcAlign + hc * hcAlign + 2 * hc + hcHc;
    uint64_t perRowBytes = perRowFloats * sizeof(float);

    uint64_t availBytes = (fixedBytes < UB_SIZE) ? (UB_SIZE - fixedBytes) : 0;
    uint64_t tileRows = availBytes / perRowBytes;
    if (tileRows < 1) tileRows = 1;
    if (tileRows > 255) tileRows = 255;

    return tileRows;
}

// ============================================================================
// Host 侧 Tiling 计算
// ============================================================================

inline void ComputeTiling(HcSplitSinkhornTiling& tiling,
    uint64_t totalBatch, uint64_t hc,
    uint32_t sinkhornIters, float eps,
    int64_t availableCoreNum,
    const float* hcScale, const float* hcBase)
{
    tiling.totalBatch = totalBatch;
    tiling.hc = hc;
    tiling.mixHc = (2 + hc) * hc;
    tiling.sinkhornIters = (sinkhornIters < 1) ? 1 : sinkhornIters;
    tiling.eps = eps;

    // 对齐计算
    tiling.mixHcAlign = align32_elements(tiling.mixHc);
    tiling.hcAlign = align32_elements(hc);

    // 拷贝参数
    for (int i = 0; i < 3; i++) tiling.hcScale[i] = hcScale[i];
    for (uint64_t i = 0; i < tiling.mixHc; i++) tiling.hcBase[i] = hcBase[i];
    for (uint64_t i = tiling.mixHc; i < HC_SPLIT_SINKHORN_MAX_MIX_HC; i++) tiling.hcBase[i] = 0.0f;

    // 多核切分
    uint64_t coreNum = static_cast<uint64_t>(availableCoreNum > 0 ? availableCoreNum : 1);
    tiling.rowsPerCore = (totalBatch + coreNum - 1) / coreNum;
    tiling.usedCoreNum = static_cast<uint32_t>((totalBatch + tiling.rowsPerCore - 1) / tiling.rowsPerCore);
    if (tiling.usedCoreNum < 1) tiling.usedCoreNum = 1;
    if (tiling.usedCoreNum > coreNum) tiling.usedCoreNum = static_cast<uint32_t>(coreNum);

    tiling.tailCoreRows = static_cast<uint32_t>(
        totalBatch - (static_cast<uint64_t>(tiling.usedCoreNum) - 1) * tiling.rowsPerCore);

    // UB 切分
    tiling.tileRows = static_cast<uint32_t>(calcTileRows(hc, tiling.mixHcAlign, tiling.hcAlign));
    tiling.tilesPerCore = static_cast<uint32_t>(
        (tiling.rowsPerCore + tiling.tileRows - 1) / tiling.tileRows);
}
