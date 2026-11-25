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

#include <cstdlib>

#include "ual/ops/reshape_and_cache/find_reshape_and_cache.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::ReshapeAndCacheArgs;

int findReshapeAndCacheBranch(const ReshapeAndCacheArgs *arg) {
    // Branch 3: blocked copy (best for PREFILL with block-contiguous
    // slot_mapping, e.g. vLLM prefill: slot = block_table[t/BS]*BS + t%BS)
    //  - one G->G memcpy per 32-token block (whole KV block contiguous on
    //    both sides) instead of per-token 32B scatter
    //  - requires num_tokens >= block_size (prefill-like) and SIMD-friendly
    //    head size (16B alignment for the block memcpy); non-contiguous
    //    blocks inside fall back to the per-token path (bit-identical)
    //  - env switch for same-build A/B: TECOOPS_RAC_BLOCKED=0 -> old
    //    selection (branch 2/1/0 as before)
    static const bool blocked_enabled = [] {
        const char *env = std::getenv("TECOOPS_RAC_BLOCKED");
        return env == nullptr || env[0] != '0';
    }();
    if (blocked_enabled && arg->size_per_head % 16 == 0 &&
        arg->num_tokens >= arg->block_size) {
        return 3;
    }
    // Branch 2: SIMD-vectorized version (best for decode, small num_tokens, all head_sizes)
    //  - 16-wide half SIMD stores eliminate head loop overhead
    //  - especially effective when head_size % 16 == 0 (most production configs)
    if (arg->size_per_head % 16 == 0) {
        return 2;
    }
    // Branch 1: merged memcpy (K+V back-to-back, fewer HBM transactions)
    //  - effective for large num_tokens (prefill) where throughput matters more
    //    than per-token latency
    if (arg->num_tokens >= 64) {
        return 1;
    }
    // Branch 0: original (default fallback)
    return 0;
}

}  // namespace ops
}  // namespace ual
}  // namespace tecoops
