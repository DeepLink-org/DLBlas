/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/* apply_mix Tiling constants and data structures (shared by Host and Kernel)
 *
 * v3.0: Manual Muls+Add with Double Buffer and bulk block load
 * - Replaces PopStackBuffer/ReduceSum RA with manual per-row Muls+Add
 * - Eliminates LCM allocation overhead, scalar_ratio target < 50%
 * - Keeps Double Buffer (TQue<2>), dynamic tiling, mix caching, blockNum clamping
 */

#pragma once

#include <cstdint>
#include <algorithm>

constexpr uint32_t DOUBLE_BUFFER = 2;
// DAV_2201 UB capacity: 192 KB = 196608 bytes
constexpr uint32_t UB_SIZE = 196608;
// Minimum tile size along A0 (64 elements for vector efficiency)
constexpr uint32_t MIN_TILE_A0 = 64;
// Maximum supported R (mhc dimension); enforced by tiling to prevent stack overflow
// of the kernel-side mixVals[] array. Typical use has R ≤ 8.
constexpr uint32_t MAX_MHC_R = 32;

// UB_OVERHEAD: accounts for TPipe queue management structures, alignment padding,
// and buffer descriptor overhead. Empirically determined for DAV_2201.
constexpr uint32_t UB_OVERHEAD = 512;

struct ApplyMixTilingData {
    uint32_t blockNum;       // Number of blocks (cores) used, ≤ actual core count
    uint32_t A1;             // Batch dimension = n0 * n1
    uint32_t R;              // Reduction dimension = mhc
    uint32_t A0;             // Feature dimension = h
    uint32_t tileA0Len;      // Tile size along A0 (64-aligned, dynamically computed from UB capacity)
    uint32_t alignedCols;    // 32B-aligned column count (in float elements)
    uint32_t totalTiles;     // Total tiles = A1 * ceil(A0 / tileA0Len)
    uint32_t tilesPerCore;   // Tiles processed per core
};

inline ApplyMixTilingData ComputeTiling(
    uint32_t n0, uint32_t n1, uint32_t mhc, uint32_t h, uint32_t coreNum)
{
    ApplyMixTilingData t;
    t.A1 = n0 * n1;
    t.R = mhc;
    t.A0 = h;

    // Safety: clamp R to MAX_MHC_R to prevent kernel stack overflow
    if (t.R > MAX_MHC_R) {
        t.R = MAX_MHC_R;
    }

    // Dynamically compute max tile size based on UB capacity:
    //   Buffer layout (all fp32, sizeof(float)=4):
    //     inQueueX: 2 * R * alignedCols * 4  =  8 * R * alignedCols  (Double Buffer)
    //     mixQ:     1 * R * 4               =  4 * R                 (Single Buffer)
    //     outQueueY:2 * alignedCols * 4     =  8 * alignedCols       (Double Buffer)
    //     overhead: UB_OVERHEAD
    //
    //  Total UB ≈ 8*(R+1)*alignedCols + 4*R + UB_OVERHEAD
    //
    //  For DAV_2201 (UB=196608):
    //    maxTileA0Len ≤ (UB_SIZE - 4*R - UB_OVERHEAD) / (8*(R+1))
    uint32_t fixedCost = 4 * t.R + UB_OVERHEAD;
    uint32_t perElementCost = 8 * (t.R + 1);  // 8*(R+1) bytes per alignedCols element

    uint32_t maxTile = (UB_SIZE >= fixedCost) ? ((UB_SIZE - fixedCost) / perElementCost) : 0;
    maxTile = (maxTile / MIN_TILE_A0) * MIN_TILE_A0;
    if (maxTile < MIN_TILE_A0) maxTile = MIN_TILE_A0;

    t.tileA0Len = std::min(maxTile, h);
    t.tileA0Len = (t.tileA0Len / MIN_TILE_A0) * MIN_TILE_A0;
    if (t.tileA0Len == 0) t.tileA0Len = MIN_TILE_A0;

    // 32B-aligned column count (in float elements)
    // Since tileA0Len is 64-aligned, tileA0Len * sizeof(float) is already 32B-aligned,
    // so alignedCols == tileA0Len in practice.
    t.alignedCols = ((t.tileA0Len * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    // Multi-core splitting: tiles along A1 and A0
    uint32_t tilesPerA0 = (h + t.tileA0Len - 1) / t.tileA0Len;
    t.totalTiles = t.A1 * tilesPerA0;

    t.tilesPerCore = (t.totalTiles + coreNum - 1) / coreNum;
    t.blockNum = (t.totalTiles + t.tilesPerCore - 1) / t.tilesPerCore;

    // blockNum must not exceed actual core count (M2 fix)
    if (t.blockNum > coreNum) t.blockNum = coreNum;

    return t;
}
