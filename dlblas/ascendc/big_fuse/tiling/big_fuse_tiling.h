/**
 * big_fuse Tiling Header
 *
 * Shared TilingHeader definitions for Kernel 1 (MatMul) and Kernel 2 (Vector).
 * This file is included by both kernel and host code.
 *
 * NOTE: TCubeTiling is defined by the CANN SDK as AscendC::tiling::TCubeTiling.
 * This header must be included AFTER the CANN headers that define TCubeTiling.
 *
 * NpuArch: DAV_2201
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)

// =============================================================================
// Kernel 0 Tiling Header (bf16->fp32 Conversion, AIV)
// =============================================================================
struct TilingHeaderK0 {
    int32_t nTokens;           // 512
    int32_t mhcMult;           // 4
    int32_t hiddenSize;        // 1280
    int32_t rgs;               // 5120
    int32_t tokensPerCore;     // ceil(512/48) = 11
    int32_t tokensPerTile;     // 4
    int32_t vecCoreNum;        // 48
    int32_t reserved[1];       // padding for 8-byte alignment
};

// Verify TilingHeaderK0 fits
static_assert(sizeof(TilingHeaderK0) <= 1024, "TilingHeaderK0 too large");

// =============================================================================
// Kernel 1 Tiling Header (MatMul)
//
// Wraps the CANN's AscendC::tiling::TCubeTiling with derived fields.
// TCubeTiling is populated by MatmulApiTiling::GetTiling().
// =============================================================================
struct TilingHeaderK1 {
    AscendC::tiling::TCubeTiling cubeTiling;  // CANN standard (~200+ bytes)
    int32_t mTotalCnt;                         // ceil(M / singleCoreM)
    int32_t nTotalCnt;                         // ceil(N / singleCoreN)
    int32_t totalBlock;                        // mTotalCnt * nTotalCnt
    int32_t mBaseTail;                         // M - (mTotalCnt - 1) * singleCoreM
    int32_t nBaseTail;                         // N - (nTotalCnt - 1) * singleCoreN
    int32_t convTileK;                         // DI-001: Phase 0 K tile size for bf16→fp32 conversion
    int32_t reserved[3];                       // padding for 8-byte alignment
};

// Verify TilingHeaderK1 fits within the TilingData constraint (single copy, <1KB)
static_assert(sizeof(TilingHeaderK1) <= 1024, "TilingHeaderK1 too large");

// =============================================================================
// Kernel 2 Tiling Header (Vector Post-process)
// =============================================================================
static constexpr int32_t SCALE_VEC_SIZE = 24;
static constexpr int32_t BASE_VEC_SIZE  = 24;

struct TilingHeaderK2 {
    int32_t nTokens;              // 512
    int32_t mhcMult;              // 4
    int32_t hiddenSize;           // 1280
    int32_t mhcMult3;             // 24
    int32_t rgs;                  // 5120
    int32_t tokensPerCore;        // ceil(512 / 48) = 11
    int32_t tokensPerTile;        // 4
    int32_t vecCoreNum;           // 48
    int32_t sinkhornRepeat;       // 10
    float   rmsEps;
    float   mhcPreEps;
    float   mhcSinkhornEps;
    float   mhcPostMultValue;
    float   scaleVec[SCALE_VEC_SIZE];  // expanded mhc_scale[3] -> [24]
    float   baseVec[BASE_VEC_SIZE];     // mhc_base[24]
    int32_t reserved[4];          // padding for alignment
};

// Verify TilingHeaderK2 fits
static_assert(sizeof(TilingHeaderK2) <= 1024, "TilingHeaderK2 too large");

#pragma pack(pop)
