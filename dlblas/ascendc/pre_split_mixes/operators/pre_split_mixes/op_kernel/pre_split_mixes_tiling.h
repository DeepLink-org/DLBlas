#pragma once

#include <cstdint>

// DAV_2201 UB 总容量: 192 KB
constexpr uint32_t UB_SIZE = 192 * 1024;

// 使用 #pragma pack 确保 ASC 编译器与 C++ 编译器对齐一致
#pragma pack(push, 8)

struct PreSplitMixesTilingData {
    // === 问题规格 ===
    int64_t totalRows;          // offset  0
    int32_t mhcMult;            // offset  8
    int32_t _pad0;              // offset 12 (padding)
    int64_t mhcMult3;           // offset 16
    float   mhcPreEps;          // offset 24
    float   mhcPostMultValue;   // offset 28

    // === 多核切分 ===
    int32_t coreNum;            // offset 32
    int32_t _pad1;              // offset 36 (padding)
    int64_t rowsPerCore;        // offset 40
    int64_t tailRows;           // offset 48

    // === UB 切分 ===
    int64_t rowsPerChunk;       // offset 56
    int64_t ubLoopPerCore;      // offset 64
    int64_t ubLoopTailCore;     // offset 72

    // === Sigmoid 临时空间 (bytes) ===
    uint32_t sigmoidTmpBufSize; // offset 80
    int32_t _pad2;              // offset 84 (padding)

    // === 输出 GM 指针 ===
    uint64_t preGmAddr;         // offset 88
    uint64_t postGmAddr;        // offset 96
    uint64_t combGmAddr;        // offset 104
};

#pragma pack(pop)