// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ual/ops/reshape_and_cache/reshape_and_cache.hpp"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/reshape_and_cache_args.h"

using tecoops::ual::args::ReshapeAndCacheArgs;
using tecoops::ual::ops::ReshapeAndCacheOp;

static tecoopsStatus_t checkReshapeAndCacheInput(tecoopsHandle_t handle,
                                                  const void *key, const void *value,
                                                  const int64_t *slot_mapping,
                                                  void *key_cache, void *value_cache,
                                                  int num_tokens, int num_kv_heads, int head_size,
                                                  int num_blocks, int block_size) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }
    if (key == nullptr || value == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (slot_mapping == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (key_cache == nullptr || value_cache == nullptr) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    if (num_kv_heads <= 0 || head_size <= 0 || num_tokens <= 0 ||
        num_blocks <= 0 || block_size <= 0) {
        return TECOOPS_STATUS_BAD_PARAM;
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsReshapeAndCache(
    tecoopsHandle_t handle,
    const void *key, const void *value,
    const int64_t *slot_mapping,
    void *key_cache, void *value_cache,
    int num_tokens, int num_kv_heads, int head_size,
    int num_blocks, int block_size) {

    tecoopsStatus_t input_error = checkReshapeAndCacheInput(
        handle, key, value, slot_mapping, key_cache, value_cache,
        num_tokens, num_kv_heads, head_size, num_blocks, block_size);
    if (input_error != TECOOPS_STATUS_SUCCESS)
        return input_error;

    // Auto-derive strides from shapes (contiguous half precision)
    constexpr int kHalfSize = 2;
    int entry_stride = head_size * kHalfSize;
    int head_stride = block_size * entry_stride;
    int block_stride = num_kv_heads * head_stride;
    int key_stride = num_kv_heads * head_size * kHalfSize;
    int value_stride = key_stride;

    ReshapeAndCacheArgs arg;
    arg.num_kv_heads = num_kv_heads;
    arg.size_per_head = head_size;
    arg.num_tokens = num_tokens;
    arg.num_blocks = num_blocks;
    arg.block_size = block_size;
    arg.block_stride = block_stride;
    arg.head_stride = head_stride;
    arg.entry_stride = entry_stride;
    arg.key_stride = key_stride;
    arg.value_stride = value_stride;
    arg.key = key;
    arg.value = value;
    arg.slot_mapping = slot_mapping;
    arg.key_cache = key_cache;
    arg.value_cache = value_cache;

    RUN_OP(ReshapeAndCacheOp, arg, arg, handle);
    return TECOOPS_STATUS_SUCCESS;
}
