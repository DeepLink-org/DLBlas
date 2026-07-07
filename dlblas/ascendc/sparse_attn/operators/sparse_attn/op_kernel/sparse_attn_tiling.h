/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// SparseAttn Tiling 数据结构 - kernel 和 host 共用
// 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字
// ============================================================================

#pragma once

#include <cstdint>

// UB 容量限制 (Ascend910B2 DAV_2201: 192KB = 196608B)
constexpr uint32_t UB_SIZE = 196608;
constexpr uint32_t UB_SAFETY = 85;  // 安全系数 85%
constexpr uint32_t UB_AVAIL = UB_SIZE * UB_SAFETY / 100;  // ~163KB
constexpr uint32_t TILE_M_MAX = 16;  // tile_m 安全上限

// Tiling 数据结构
struct SparseAttnTiling {
    // Shape 参数
    uint32_t b;       // batch size
    uint32_t m;       // query sequence length
    uint32_t n;       // KV sequence length
    uint32_t h;       // number of heads
    uint32_t d;       // head dimension
    uint32_t topk;    // sparse attention window size

    // 多核切分
    uint32_t totalTasks;   // = b * m
    uint32_t usedCoreNum;  // = min(aivNum, totalTasks)
    uint32_t tasksPerCore; // = CeilDiv(totalTasks, usedCoreNum)

    // 每个 task 的 tile 信息
    uint32_t tile_m;       // 单次迭代处理的 query position 数

    // 每个 core 的 task 信息
    uint32_t coreTaskStart; // 当前 core 处理的第一个 task idx
    uint32_t coreTaskCount; // 当前 core 处理的 task 数量

    // 计算参数
    float softmax_scale;
};

// Tiling 计算函数声明（供 host 侧使用）
// 由外部提供 platform_ascendc::PlatformAscendC 计算
