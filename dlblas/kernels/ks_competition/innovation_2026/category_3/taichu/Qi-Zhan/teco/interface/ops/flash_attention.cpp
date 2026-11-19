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

#include <sdaa_driver_api.h>
#include <sdaa_runtime.h>

#include "interface/common/convert.h"
#include "interface/common/macro.h"
#include "interface/include/builtin_type.h"
#include "interface/include/tecoops.h"
#include "ual/args/flash_attention_args.h"

using tecoops::Convert;
using tecoops::ual::args::FlashAttentionArgs;

namespace sdaa {

class CommonData {
 public:
    static CommonData *instance();
    int find_device_function(int device_id, const void *host_function,
                             SDfunction *device_function);
};

}  // namespace sdaa

namespace {

struct PendingDecodeAppend {
    bool valid = false;
    const void *key = nullptr;
    const void *value = nullptr;
    const int64_t *slot_mapping = nullptr;
    int key_head_stride = 0;
    int value_head_stride = 0;
};

thread_local PendingDecodeAppend g_pending_decode_append;
thread_local SDfunction g_decode_m64_function;

}  // namespace

extern "C" void tecoopsSetPendingDecodeAppend(
    const void *key, const void *value, const int64_t *slot_mapping,
    int key_head_stride, int value_head_stride) {
    g_pending_decode_append.valid = true;
    g_pending_decode_append.key = key;
    g_pending_decode_append.value = value;
    g_pending_decode_append.slot_mapping = slot_mapping;
    g_pending_decode_append.key_head_stride = key_head_stride;
    g_pending_decode_append.value_head_stride = value_head_stride;
}

extern "C" tecoopsStatus_t tecoopsFlashAttentionDecodeM64Fused(
    tecoopsHandle_t handle,
    int max_seqlen_k, int max_block_num,
    const int *kv_seq_lens,
    const int *block_table, int block_table_dim,
    const void *q, int q_token_stride, int q_head_stride,
    void *key_cache, void *value_cache,
    void *out, int out_token_stride, int out_head_stride,
    const void *append_key, const void *append_value,
    const int64_t *append_slot_mapping,
    int append_key_head_stride, int append_value_head_stride) {
    if (handle == nullptr) {
        return TECOOPS_STATUS_NOT_INITIALIZED;
    }

    FlashAttentionArgs args{};
    args.spe_num = handle->spe_num;
    args.batch_size = 1;
    args.size_per_head = 64;
    args.local_head_num = 4;
    args.local_kv_head_num = 2;
    args.block_size = 32;
    args.block_table_dim = block_table_dim;
    args.q_token_stride = q_token_stride;
    args.q_head_stride = q_head_stride;
    args.out_token_stride = out_token_stride;
    args.out_head_stride = out_head_stride;
    args.max_q_seq_len = 1;
    args.max_k_seq_len = max_seqlen_k;
    args.max_block_num = max_block_num;
    args.softmax_scale = 0.125f;
    args.kv_seq_lens = kv_seq_lens;
    args.block_table = block_table;
    args.q = q;
    args.key_cache = key_cache;
    args.value_cache = value_cache;
    args.fmha_out = out;

    args.workspace = nullptr;
    args.append_pending = 1;
    args.append_key = append_key;
    args.append_value = append_value;
    args.append_slot_mapping = append_slot_mapping;
    args.append_key_head_stride = append_key_head_stride;
    args.append_value_head_stride = append_value_head_stride;

    void *kernel_args[] = {
        SDAA_LAUNCH_PARAM_BUFFER_POINTER,
        &args,
        SDAA_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void *>(sizeof(FlashAttentionArgs)),
        SDAA_LAUNCH_PARAM_END,
    };
    if (g_decode_m64_function == nullptr) {
        int device_id = 0;
        sdaaError_t device_status = sdaaGetDevice(&device_id);
        if (device_status != sdaaSuccess ||
            sdaa::CommonData::instance()->find_device_function(
                device_id,
                reinterpret_cast<const void *>(
                    teco_slave_flash_attention_decode_m64),
                &g_decode_m64_function) != 0 ||
            g_decode_m64_function == nullptr) {
            return TECOOPS_STATUS_INTERNAL_ERROR;
        }
    }
    SDresult launch_status = sdLaunchKernel(
        g_decode_m64_function,
        reinterpret_cast<SDstream>(handle->stream), nullptr, kernel_args);
    return launch_status == SDAA_SUCCESS
        ? TECOOPS_STATUS_SUCCESS
        : TECOOPS_STATUS_INTERNAL_ERROR;
}

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
    args.q_token_stride = qDataDesc->strideA[0];
    args.q_head_stride = qDataDesc->strideA[1];
    args.out_token_stride = oDataDesc->strideA[0];
    args.out_head_stride = oDataDesc->strideA[1];
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
    args.workspace = workspace;
    args.append_pending = g_pending_decode_append.valid ? 1 : 0;
    args.append_key = g_pending_decode_append.key;
    args.append_value = g_pending_decode_append.value;
    args.append_slot_mapping = g_pending_decode_append.slot_mapping;
    args.append_key_head_stride =
        g_pending_decode_append.key_head_stride;
    args.append_value_head_stride =
        g_pending_decode_append.value_head_stride;
    g_pending_decode_append = PendingDecodeAppend{};

    bool use_m64_decode = args.batch_size == 1 &&
                          args.max_q_seq_len == 1 &&
                          args.local_head_num == 4 &&
                          args.local_kv_head_num == 2 &&
                          args.size_per_head == 64 &&
                          args.block_size == 32 &&
                          args.spe_num == 32;
    bool use_m64_prefill = args.batch_size == 1 &&
                           args.max_q_seq_len > 1 &&
                           args.max_q_seq_len <= args.max_k_seq_len &&
                           args.local_head_num == 4 &&
                           args.local_kv_head_num == 2 &&
                           args.size_per_head == 64 &&
                           args.block_size == 32 &&
                           args.spe_num == 32;
    if (use_m64_decode) {
        RUN_KERNEL(teco_slave_flash_attention_decode_m64,
                   handle->stream, args);
    } else if (use_m64_prefill) {
        RUN_KERNEL(teco_slave_flash_attention_prefill_m64,
                   handle->stream, args);
    } else {
        ERROR("unsupported Flash Attention shape without generic fallback: "
              "batch_size=%d max_q_seq_len=%d max_k_seq_len=%d "
              "heads=%d kv_heads=%d head_dim=%d block_size=%d spe_num=%d\n",
              args.batch_size, args.max_q_seq_len, args.max_k_seq_len,
              args.local_head_num, args.local_kv_head_num,
              args.size_per_head, args.block_size, args.spe_num);
        return TECOOPS_STATUS_NOT_SUPPORTED;
    }

    return TECOOPS_STATUS_SUCCESS;
}
