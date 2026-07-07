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
// expand_kenel_fwd Tiling 常量和结构体 (kernel 和 host 共用)
// ============================================================================
//
// 本文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字。
// 所有编译单元（kernel、host main、torch host）共用此文件。
// ============================================================================

#pragma once

#include <cstdint>

// UB 容量约 192KB，预留 4KB 给队列头/对齐额外开销
constexpr uint32_t UB_BUDGET_BYTES = 188 * 1024;  // 192512 bytes
constexpr uint32_t DOUBLE_BUFFER = 2;

// DataCopy/DataCopyPad 的最小对齐要求：16 元素 = 32 字节
constexpr int64_t UB_ALIGN_ELEMS = 16;
constexpr int64_t UB_ALIGN_BYTES = 32;

// Expand Tiling 参数结构体
struct ExpandTilingData {
    int64_t totalRows;      // 展平后总行数 = B * S
    int64_t H;              // hidden_dim (最后一维大小)
    int64_t M;              // mhc_mult 扩展倍数
    int64_t tileH;          // UB buffer 中 tileH 大小 (对齐到 16 倍数，>= H)
    int64_t rowsPerCore;    // 每个 AI Core 处理的行数
    int64_t usedCoreCnt;    // 实际使用的 AI Core 数量
    int64_t totalTiles;     // 总 tile 数 (始终为 totalRows，无 H 维切分)
    int64_t tailH;          // 同 H (始终为 H，无 H 维切分)
    uint32_t dtypeSize;     // sizeof(T) in bytes (2=FP16/BF16, 4=FP32)
};
