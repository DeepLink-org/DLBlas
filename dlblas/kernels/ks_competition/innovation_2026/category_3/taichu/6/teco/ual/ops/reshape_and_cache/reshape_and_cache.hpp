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

#ifndef TECO_UAL_OPS_RESHAPE_AND_CACHE_RESHAPE_AND_CACHE_HPP_
#define TECO_UAL_OPS_RESHAPE_AND_CACHE_RESHAPE_AND_CACHE_HPP_

#include "ual/ops/base_op.hpp"
#include "ual/com/log.h"
#include "ual/ops/reshape_and_cache/find_reshape_and_cache.h"
#include "ual/kernel/reshape_and_cache/reshape_and_cache.h"

namespace tecoops {
namespace ual {
namespace ops {

using tecoops::ual::args::ReshapeAndCacheArgs;
using tecoops::ual::common::Status;

struct ReshapeAndCacheType {
    using ArgsType = ReshapeAndCacheArgs;
    using PatchType = ReshapeAndCacheArgs;
    using RetType = void;
    using PImplType = void (*)(ArgsType);
};

static ReshapeAndCacheType::PImplType ReshapeAndCacheAlgos[] = {
    teco_slave_reshape_and_cache_fp16,
    teco_slave_reshape_and_cache_merged_fp16,
    teco_slave_reshape_and_cache_simd_fp16,
    teco_slave_reshape_and_cache_blocked_fp16,
};

static const char *ReshapeAndCacheDiscription[] = {
    "teco_slave_reshape_and_cache_fp16",
    "teco_slave_reshape_and_cache_merged_fp16",
    "teco_slave_reshape_and_cache_simd_fp16",
    "teco_slave_reshape_and_cache_blocked_fp16",
};

struct ReshapeAndCacheOp : public BaseOp<ReshapeAndCacheOp, ReshapeAndCacheType> {
 public:
    using ArgsType = typename ReshapeAndCacheType::ArgsType;
    using PatchType = typename ReshapeAndCacheType::PatchType;
    using RetType = typename ReshapeAndCacheType::RetType;
    using PImplType = typename ReshapeAndCacheType::PImplType;

    ReshapeAndCacheOp() = default;
    ~ReshapeAndCacheOp() = default;

    static const char *name() { return "reshape_and_cache"; }

    Status findImpl(const PatchType *args) {
        int index = findReshapeAndCacheBranch(args);
        if (index == -1) {
            ERROR("reshape_and_cache branch is not exit!");
            return Status::NOT_IMPLEMENTED;
        }
        setInstance(ReshapeAndCacheAlgos[index], ReshapeAndCacheDiscription[index]);
        return Status::SUCCESS;
    }
};

}  // namespace ops
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_OPS_RESHAPE_AND_CACHE_RESHAPE_AND_CACHE_HPP_
