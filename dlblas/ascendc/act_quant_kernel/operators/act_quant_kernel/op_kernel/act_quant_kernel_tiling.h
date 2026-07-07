/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// act_quant_kernel Tiling 常量和结构体 - kernel 和 host 共用
// ============================================================================
//
// 算子: Activation Per-Group FP8 Quantization
// 输入: x (bf16/fp16, [..., N]) -> 输出: x_q (fp8_e4m3fn) + x_s (fp32)
//
// Tiling 策略: 按 groups 维度切分到多核，每核内按 tileGroups 分批

#pragma once

#include <cstdint>

// [CONFIG] 硬件约束
constexpr uint32_t UB_SIZE = 196608;         // 192 KB (DAV_2201)
constexpr uint32_t DOUBLE_BUFFER = 2;

// [CONFIG] FP8_e4m3fn 数值范围
constexpr float FP8_E4M3FN_MAX = 448.0f;
constexpr float FP8_E4M3FN_MIN = -448.0f;

// [CONFIG] ReduceMax work buffer 大小
constexpr uint32_t REDUCE_BUF_SIZE = 32 * 1024;

// [CONFIG] Input dtype enumeration
enum class InputDtype : uint32_t {
    BF16 = 0,
    FP16 = 1
};

// Tiling 数据结构 - 向核函数传递的运行时参数
struct ActQuantTiling {
    uint32_t numGroups;          // total groups = totalElements / groupSize
    uint32_t groupSize;          // elements per group (G)
    uint32_t groupSizeAlign;     // 32B-aligned G (in elements)
    uint32_t tileGroups;         // groups processed per tile
    uint32_t coreGroups;         // groups assigned per core
    float    fp8Max;             // fp8_e4m3fn max (448.0)
    float    fp8Min;             // fp8_e4m3fn min (-448.0)
    float    eps;                // amax lower clamp
    bool     scaleUe8m0;         // UE8M0 scale format
    InputDtype inputDtype;       // input dtype enum
};

// Calculate 32B-aligned group size in elements
inline uint32_t calcGroupSizeAlign(uint32_t groupSize, uint32_t dsize) {
    uint32_t bytes = groupSize * dsize;
    uint32_t bytesAlign = ((bytes + 31) / 32) * 32;
    return bytesAlign / dsize;
}
