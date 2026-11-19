// BSD 3-Clause License
//
// Copyright (c) 2024, Tecorigin Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
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

#include "ual/ops/flash_attention/flash_attention.hpp"

#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/flash_attention_args.h"

using tecoops::Convert;
using tecoops::ual::args::FlashAttentionArgs;
using tecoops::ual::args::FlashAttentionPatchArgs;
using tecoops::ual::ops::FlashAttentionOp;

// check the compliance of the args of reduce variance operator
static tecoopsStatus_t flashAttentionCheckArgs(tecoopsHandle_t handle,
                                               const tecoopsTensorDescriptor_t kCacheDesc,
                                               const tecoopsTensorDescriptor_t vCacheDesc) {
    if (handle == nullptr) {
        ERROR("NULL Error: tecoopsHandle_t is nullptr \n");
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }

    if (kCacheDesc->dataType != vCacheDesc->dataType) {
        ERROR("k_cache data_type must be equal to v_cache data_type \n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    if (kCacheDesc->dataType != TECOOPS_DATA_HALF) {
        ERROR("k v cache can only support TECOOPS_DATA_HALF now\n");
        return TECOOPS_STATUS_BAD_PARAM;
    }

    for (int i = 0; i < kCacheDesc->nbDims; ++i) {
        if (kCacheDesc->dimA[i] != vCacheDesc->dimA[i]) {
            ERROR("shape of k v cache is not equal\n");
            return TECOOPS_STATUS_BAD_PARAM;
        }
    }
    return TECOOPS_STATUS_SUCCESS;
}

tecoopsStatus_t tecoopsFlashAttention(tecoopsHandle_t handle,
                                      int max_seqlen_q, int max_seqlen_k,
                                      int max_block_num, const int *q_seq_lens,
                                      const int *kv_seq_lens, 
                                      const tecoopsTensorDescriptor_t blockTableDesc,
                                      const void *blockTable,
                                      const tecoopsTensorDescriptor_t qDataDesc,
                                      const void *qData,
                                      const tecoopsTensorDescriptor_t kCacheDesc,
                                      const void *kCache,
                                      const tecoopsTensorDescriptor_t vCacheDesc,
                                      const void *vCache,
                                      const tecoopsTensorDescriptor_t oDataDesc,
                                      void *oData, void *workspace) {
    auto input_error = flashAttentionCheckArgs(handle, kCacheDesc, vCacheDesc);
    checkTecoopsStatus(input_error);
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }

    FlashAttentionArgs args;
    args.spe_num = handle->spe_num;
    args.batch_size = blockTableDesc->dimA[0];
    args.size_per_head = qDataDesc->dimA[2];
    args.local_head_num = qDataDesc->dimA[1];
    args.local_kv_head_num = kCacheDesc->dimA[1];
    args.block_size = kCacheDesc->dimA[2];
    args.block_table_dim = blockTableDesc->dimA[1];
    args.max_q_seq_len = max_seqlen_q;              // useless
    args.max_k_seq_len = max_seqlen_k;
    args.max_block_num = kCacheDesc->dimA[0];
    args.softmax_scale = 1.0 / sqrtf(double(args.size_per_head));
    args.q_seq_lens = q_seq_lens;
    args.kv_seq_lens = kv_seq_lens;
    args.block_table = (int *)blockTable;
    args.q = qData;
    args.key_cache = (void *)kCache;
    args.value_cache = (void *)vCache;
    args.fmha_out = oData;
    args.workspace = workspace;                     // nullptr

    FlashAttentionPatchArgs patch_arg;
    patch_arg.rvargs = &args;
    patch_arg.data_type = tecoops::ual::common::UAL_DTYPE_HALF;

    RUN_OP(FlashAttentionOp, args, patch_arg, handle);

    return TECOOPS_STATUS_SUCCESS;
}