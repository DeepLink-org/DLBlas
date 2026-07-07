/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#pragma once

#include <cstdint>

// Tiling constants
constexpr uint32_t DOUBLE_BUFFER = 2;

struct HeadComputeMixBwdTiling {
    // Multi-core split
    uint32_t total_rows;
    uint32_t inner_dim;
    uint32_t core_num;
    uint32_t rows_per_core;
    uint32_t block_num;
    uint32_t tail_rows;

    // UB split
    uint32_t tile_rows;
    uint32_t ub_loops;

    // Sigmoid tmpBuf
    uint32_t sigmoid_tmp_size;

    // Workspace (group reduce)
    uint32_t workspace_size;

    // Per-core workspace offset stride (aligned to 256 bytes)
    uint32_t ws_offset_stride;
};
