/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// Tiling constants and data structure for engram_fused_weight.
// Shared between kernel (.asc) and host.
// Per DESIGN.md §5.4 and §9.4.

#pragma once

#include <cstdint>
#include <algorithm>

// Single-buffer depth (standard pattern, avoids ping-pong sync complexity)
constexpr uint32_t QUE_DEPTH = 1;

// Minimum tiling data per AI Core: 4KB in bits (DESIGN.md §5.1)
constexpr int64_t MIN_TILING_BITS_SIZE_PER_CORE = 32768;

// Element alignment factor: 512 elements (DESIGN.md §5.1 Step 2)
constexpr int64_t ELEM_ALIGN_FACTOR = 512;

// 256-byte alignment for UB chunks
constexpr int64_t ALIGN_256 = 256;

// UB capacity for DAV_2201: 192 KB (DESIGN.md §5.2)
constexpr int64_t UB_SIZE = 192 * 1024;

// Per DESIGN.md §5.4: Tiling parameter structure
struct EngramFusedWeightTilingData {
    int64_t dim0;           // Total element count (hc_mult * hidden_size)
    int32_t coreNum;        // Number of AI Cores used
    int64_t blockFormer;    // Elements per block (per core)
    int64_t blockNum;       // Number of blocks
    int64_t ubFormer;       // UB chunk size in elements
    int64_t ubLoop;         // Number of UB loops per block
    int64_t ubTail;         // Tail elements for last UB loop
};

// Per DESIGN.md §9.4: Host-side Tiling computation
inline EngramFusedWeightTilingData ComputeTiling(
    int64_t hc_mult, int64_t hidden_size, int32_t availableCoreNum)
{
    constexpr int64_t MIN_TILING_BITS = 32768;   // 4KB
    constexpr int64_t ELEM_ALIGN = 512;
    constexpr int64_t ALIGN_256B = 256;

    int64_t dim0 = hc_mult * hidden_size;
    if (dim0 == 0) {
        return {0, 0, 0, 0, 0, 0, 0};
    }

    // Step 1: Core count - always use single core (DESIGN.md §5.1 conclusion)
    // For this elementwise operator, data sizes are small and single-core
    // with multi-tile loop (ubLoop) is sufficient. Multi-block launch has
    // known issues on this platform for small data, so force coreNum=1.
    int32_t coreNum = 1;

    // Step 2: Block per core = all data, aligned to 512 elements
    int64_t blockFormer = ((dim0 + ELEM_ALIGN - 1) / ELEM_ALIGN) * ELEM_ALIGN;
    int64_t blockNum = 1;

    // Step 3: UB chunk calculation with mixed-precision bufferDivisor
    // Per DESIGN.md §5.2:
    //   whQue (double-buffer BF16):  2*(2) = 4 bytes/elem
    //   weQue (double-buffer BF16):  2*(2) = 4 bytes/elem
    //   tmpWH (FP32):                1*4    = 4 bytes/elem
    //   tmpWE (FP32):                1*4    = 4 bytes/elem
    //   outQue (double-buffer FP32): 2*4    = 8 bytes/elem
    //   bufferDivisor = 4+4+4+4+8 = 24 bytes/elem
    //
    // Safe ubFormer limit that works across all launch paths
    // (direct invoke <<<>>> and PyTorch function call).
    constexpr int64_t UB_FORMER_MAX = 2048;
    int64_t bufferDivisor = 2*2 + 2*2 + 4 + 4 + 2*4;  // 24
    int64_t ubSize = 192 * 1024;  // 192KB
    int64_t maxElemNum = ubSize * 8 / bufferDivisor;

    // Align to 256 bytes, BF16 input -> alignFactor = 256/2 = 128 elements
    int64_t alignFactor = ALIGN_256B * 8 / 2;  // 128
    int64_t ubFormer = (maxElemNum / alignFactor) * alignFactor;
    ubFormer = std::min(ubFormer, blockFormer);
    ubFormer = std::min(ubFormer, UB_FORMER_MAX);

    int64_t ubLoop = (blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubTail = blockFormer - (ubLoop - 1) * ubFormer;

    return {dim0, coreNum, blockFormer, blockNum, ubFormer, ubLoop, ubTail};
}
