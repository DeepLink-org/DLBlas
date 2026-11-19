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

#ifndef TECO_UAL_ARGS_RESHAPE_AND_CACHE_ARGS_H_
#define TECO_UAL_ARGS_RESHAPE_AND_CACHE_ARGS_H_

#include <cstdint>

namespace tecoops {
namespace ual {
namespace args {

// reshape_and_cache: 将 key/value 按 slot_mapping 写入 paged KV cache
// key:    [num_tokens, num_kv_heads, head_size]
// value:  [num_tokens, num_kv_heads, head_size]
// cache:  [num_blocks, num_kv_heads, block_size, head_size]
struct ReshapeAndCacheArgs {
    int num_kv_heads;
    int size_per_head;
    int num_tokens;
    int num_blocks;
    int block_size;
    int block_stride;   // bytes between blocks in cache
    int head_stride;    // bytes between heads in cache
    int entry_stride;   // bytes between entries in cache
    int key_stride;     // bytes between tokens in key input
    int value_stride;   // bytes between tokens in value input
    int bsmem;          // rac5: full-block memcpy fast path (H==1 only; TECOOPS_RAC_BSMEM)
    int racspm;         // rac6: SPM-staged full-block fast path (TECOOPS_RAC_SPM)
    int diag_mode;      // diag timing stubs 1..6 (WRONG results; TECOOPS_RAC_DIAG)
    const void *key;
    const void *value;
    const int64_t *slot_mapping;
    void *key_cache;
    void *value_cache;
};

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_ARGS_RESHAPE_AND_CACHE_ARGS_H_
