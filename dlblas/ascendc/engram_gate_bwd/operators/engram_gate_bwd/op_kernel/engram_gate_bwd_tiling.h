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
// Tiling 常量和数据结构 - kernel 和 host 共用
// 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字
// ============================================================================

#pragma once

#include <cstdint>

constexpr uint32_t DOUBLE_BUFFER = 2;

// Tiling 数据结构 - 向核函数传递的运行时参数
struct EngramGateBwdTiling {
    // 维度参数
    uint64_t totalT;          // 总序列长度
    uint64_t totalH;          // 头数
    uint64_t totalD;          // 隐层维度
    uint64_t D_align;         // D 的 32B 对齐值 (f32 下)

    // 多核切分
    uint64_t tileT;           // 每核 T 元素数 = ceil(totalT / coreNum)
    uint64_t coreNum;         // 实际使用核数
    uint64_t coreIdx;         // 当前核索引

    // UB 切分
    uint64_t tileTPerLoop;    // 每轮 UB 处理的 T 元素数
    uint64_t tailT;           // 尾轮 T 元素数 (可能 < tileTPerLoop)
    uint64_t loopCount;       // T 循环次数

    // 常量
    float clampValue;         // clamp 下限
    float eps;                // 数值稳定常数
    float scalar;             // D^{-0.5}
    float invD;               // 1.0 / D
    float half;               // 0.5
    float one;                // 1.0

    // Workspace
    uint64_t workspaceSize;   // coreNum * 2 * H * D * sizeof(float)
    uint64_t workspaceOffset; // 当前核 partial 在 workspace 中的偏移
};
