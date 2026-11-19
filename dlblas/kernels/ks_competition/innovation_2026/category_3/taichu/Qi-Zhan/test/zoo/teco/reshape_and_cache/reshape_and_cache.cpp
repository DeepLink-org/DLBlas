// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "zoo/teco/convert.h"
#include "common/time.hpp"
#include "zoo/teco/reshape_and_cache/reshape_and_cache.h"
#include "interface/include/tecoops.h"

namespace optest {

void ReshapeAndCacheExecutor::paramCheck() {
    // 输入: key, value (tensor), slot_mapping (tensor)
    // 输出: key_cache, value_cache (tensor, reused)
    if (parser_->inputs().size() != 3) {
        ALLOG(ERROR) << "reshape_and_cache requires 3 inputs: key, value, slot_mapping.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
    if (parser_->outputs().size() != 2) {
        ALLOG(ERROR) << "reshape_and_cache requires 2 outputs: key_cache, value_cache.";
        throw std::invalid_argument(std::string(__FILE__) + ":" + std::to_string(__LINE__));  // NOLINT
    }
}

void ReshapeAndCacheExecutor::paramParse() {
    auto meta_key = parser_->input(0);
    auto meta_cache = parser_->output(0);

    // 从 prototxt shape 解析参数
    // key shape: [num_tokens, num_kv_heads, head_size]
    num_tokens_    = meta_key->shape[0];
    num_kv_heads_  = meta_key->shape[1];
    head_size_     = meta_key->shape[2];

    // key_cache shape: [num_blocks, num_kv_heads, block_size, head_size]
    num_blocks_ = meta_cache->shape[0];
    block_size_ = meta_cache->shape[2];
}

void ReshapeAndCacheExecutor::paramGeneration() {
    // input tensors
    key_           = dev_input[0];
    value_         = dev_input[1];
    slot_mapping_  = static_cast<int64_t *>(dev_input[2]);

    // output tensors (reused)
    key_cache_   = dev_output[0];
    value_cache_ = dev_output[1];

    // 确保 slot_mapping 无碰撞：用顺序值 [0, 1, ..., num_tokens-1] 覆盖随机值
    // 避免 32 SPE 并行写入同一 slot 时产生 race condition
    std::vector<int64_t> unique_slots(num_tokens_);
    for (int t = 0; t < num_tokens_; t++) {
        unique_slots[t] = t;
    }
    sdaaMemcpy(slot_mapping_, unique_slots.data(),
               num_tokens_ * sizeof(int64_t), sdaaMemcpyHostToDevice);

    int cache_size = num_blocks_ * num_kv_heads_ * block_size_ * head_size_ * sizeof(short);
    sdaaMemset(key_cache_, 0, cache_size);
    sdaaMemset(value_cache_, 0, cache_size);
}

void ReshapeAndCacheExecutor::compute() {


    checkTECOOPS(tecoopsReshapeAndCache(
        handle_,
        key_, value_,
        slot_mapping_,
        key_cache_, value_cache_,
        num_tokens_, num_kv_heads_, head_size_,
        num_blocks_, block_size_));
}

int64_t ReshapeAndCacheExecutor::getTheoryOps() {
    return 0;  // 纯访存算子，无计算量
}

int64_t ReshapeAndCacheExecutor::getTheoryIoSize() {
    // 读: key + value + slot_mapping (half = 2 bytes)
    // 写: key_cache + value_cache (half = 2 bytes)
    constexpr int kHalfSize = 2;
    int64_t key_val_size = num_tokens_ * num_kv_heads_ * head_size_ * kHalfSize;
    int64_t slot_size = num_tokens_ * sizeof(int64_t);
    return key_val_size * 2 + slot_size + key_val_size * 2;
}

void ReshapeAndCacheExecutor::cpuCompute() {
    auto *key_baseline   = static_cast<short *>(baseline_input[0]);
    auto *value_baseline = static_cast<short *>(baseline_input[1]);
    auto *slot_baseline  = static_cast<int64_t *>(baseline_input[2]);
    auto *kc_baseline    = static_cast<short *>(baseline_output[0]);
    auto *vc_baseline    = static_cast<short *>(baseline_output[1]);

    int H = num_kv_heads_;
    int D = head_size_;
    int BS = block_size_;

    // 确保 slot_mapping 无碰撞（与 paramGeneration 保持一致）
    for (int t = 0; t < num_tokens_; t++) {
        slot_baseline[t] = t;
    }

    // baseline_output 初始化为 0（避免未初始化值触发 NaN 警告）
    memset(kc_baseline, 0, num_blocks_ * H * BS * D * sizeof(short));
    memset(vc_baseline, 0, num_blocks_ * H * BS * D * sizeof(short));

    for (int t = 0; t < num_tokens_; t++) {
        int64_t slot = slot_baseline[t];
        if (slot < 0) continue;
        int blk = slot / BS;
        int off = slot % BS;
        for (int h = 0; h < H; h++) {
            int src_idx = t * H * D + h * D;
            int dst_idx = blk * H * BS * D + h * BS * D + off * D;
            memcpy(kc_baseline + dst_idx, key_baseline + src_idx, D * sizeof(short));
            memcpy(vc_baseline + dst_idx, value_baseline + src_idx, D * sizeof(short));
        }
    }
}

}  // namespace optest
