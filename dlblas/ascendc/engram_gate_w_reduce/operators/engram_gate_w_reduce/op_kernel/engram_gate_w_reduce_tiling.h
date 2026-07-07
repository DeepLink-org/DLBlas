/**
 * engram_gate_w_reduce Tiling 常量和结构体
 * Kernel 和 Host 共用，只含纯 C/C++ 语法
 */

#pragma once

#include <cstdint>

// 归约维度大小固定为 108
constexpr uint32_t R_DIM = 108;
// A0 维度中的通道数（weight 的第0维大小）
constexpr uint32_t N_CHANNELS = 4;
// Double Buffer 数量
constexpr uint32_t DOUBLE_BUFFER = 2;
// UB Buffer 最大容量检查: tileA0MaxLen = 192KB / 24B = 8192 (Phase 2 限制)
constexpr uint32_t UB_MAX_TILE_A0_LEN = 8192;

struct EngramGateWReduceTiling {
    uint32_t blockDim;         // 使用的核数
    uint32_t hiddenSize;       // hidden_size
    uint32_t tileHiddenLen;    // 每个核处理的 hidden 段长度（非尾部核）
    uint32_t tileA0Len;        // 每核处理的 A0 长度 = tileHiddenLen * 4（非尾部核）
    uint32_t tailHiddenLen;    // 尾部核处理的 hidden 段长度
    uint32_t tailA0Len;        // 尾部核处理的 A0 长度
    uint32_t R;                // 归约维度大小 = 108
};
