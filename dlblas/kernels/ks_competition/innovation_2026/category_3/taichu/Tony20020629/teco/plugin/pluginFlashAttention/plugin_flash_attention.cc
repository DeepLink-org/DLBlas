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

#include <plugin/register_op.h>
#include <sdaa_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "interface/include/tecoops.h"

namespace TECO_INFER {

struct plugin_flash_attentionAttrs
    : public tvm::AttrsNode<plugin_flash_attentionAttrs> {
  int64_t max_seqlen_q;
  int64_t max_seqlen_k;
  int64_t max_block_num;
  TVM_DECLARE_ATTRS(plugin_flash_attentionAttrs,
                     "relay.attrs.plugin_flash_attentionAttrs") {
    TVM_ATTR_FIELD(max_seqlen_q).set_default(0).describe("Max Q length");
    TVM_ATTR_FIELD(max_seqlen_k).set_default(0).describe("Max K length");
    TVM_ATTR_FIELD(max_block_num).set_default(0).describe("Max number of blocks in cache");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_flash_attentionAttrs);

class PluginFlashAttentionImpl : public AbstractPluginOp {
 public:
  PluginFlashAttentionImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    // output shape = q shape: [total_tokens, num_heads, head_size]
    total_output_shape[0] = total_input_shape[0];
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    std::cout << "CALL PluginFlashAttention enqueue" << std::endl;
    void* q_dev = ctx->GetInputDataPtr("q");
    void* k_cache_dev = ctx->GetInputDataPtr("k_cache");
    void* v_cache_dev = ctx->GetInputDataPtr("v_cache");
    void* block_table_dev = ctx->GetInputDataPtr("block_table");
    void* cu_seqlens_q_dev = ctx->GetInputDataPtr("cu_seqlens_q");
    void* seqused_k_dev = ctx->GetInputDataPtr("seqused_k");
    void* out_dev = ctx->GetOutputDataPtr(0);

    int64_t max_seqlen_q, max_seqlen_k, max_block_num;
    ctx->GetAttr("max_seqlen_q", max_seqlen_q);
    ctx->GetAttr("max_seqlen_k", max_seqlen_k);
    ctx->GetAttr("max_block_num", max_block_num);

    std::vector<int> q_shape, k_cache_shape, v_cache_shape, bt_shape;
    ctx->GetInputShape("q", q_shape);           // [total_tokens, num_heads, head_size]
    ctx->GetInputShape("k_cache", k_cache_shape); // [num_blocks, kv_heads, block_size, head_size]
    (void)v_cache_shape;

    sdaaStream_t stream = ctx->GetStream();

    // Build tensor descriptors from shapes
    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    auto make_desc = [&](tecoopsTensorDescriptor_t &desc, tecoopsDataType_t dtype,
                         const std::vector<int> &shape) {
      tecoopsCreateTensorDescriptor(&desc);
      tecoopsSetTensorNdDescriptor(desc, dtype, shape.size(), shape.data(), nullptr);
    };

    tecoopsTensorDescriptor_t blockTableDesc, qDataDesc, kCacheDesc, vCacheDesc, oDataDesc;
    make_desc(blockTableDesc, TECOOPS_DATA_INT32, bt_shape);
    make_desc(qDataDesc, TECOOPS_DATA_HALF, q_shape);
    make_desc(kCacheDesc, TECOOPS_DATA_HALF, k_cache_shape);
    make_desc(vCacheDesc, TECOOPS_DATA_HALF, k_cache_shape);
    make_desc(oDataDesc, TECOOPS_DATA_HALF, q_shape);

    tecoopsFlashAttention(handle,
                          static_cast<int>(max_seqlen_q),
                          static_cast<int>(max_seqlen_k),
                          static_cast<int>(max_block_num),
                          static_cast<const int*>(cu_seqlens_q_dev),
                          static_cast<const int*>(seqused_k_dev),
                          blockTableDesc, static_cast<const void*>(block_table_dev),
                          qDataDesc, static_cast<const void*>(q_dev),
                          kCacheDesc, static_cast<const void*>(k_cache_dev),
                          vCacheDesc, static_cast<const void*>(v_cache_dev),
                          oDataDesc, out_dev,
                          /*workspace=*/nullptr);

    tecoopsDestroyTensorDescriptor(blockTableDesc);
    tecoopsDestroyTensorDescriptor(qDataDesc);
    tecoopsDestroyTensorDescriptor(kCacheDesc);
    tecoopsDestroyTensorDescriptor(vCacheDesc);
    tecoopsDestroyTensorDescriptor(oDataDesc);

    tecoopsDestroy(handle);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_flash_attention, PluginFlashAttentionImpl)

PLUGIN_REGISTER_OP("plugin_flash_attention")
    .Input("q")
    .Type("Tensor")
    .Desc("Query: [total_tokens, num_heads, head_size]")
    .Input("k_cache")
    .Type("Tensor")
    .Desc("K cache: [num_blocks, kv_heads, block_size, head_size]")
    .Input("v_cache")
    .Type("Tensor")
    .Desc("V cache: [num_blocks, kv_heads, block_size, head_size]")
    .Input("block_table")
    .Type("Tensor")
    .Desc("Block table: [batch_size, block_table_dim]")
    .Input("cu_seqlens_q")
    .Type("Tensor")
    .Desc("Cumulative Q seq lens: [batch_size + 1]")
    .Input("seqused_k")
    .Type("Tensor")
    .Desc("KV seq used per batch: [batch_size]")
    .AttrType<plugin_flash_attentionAttrs>()
    .Register();

}  // namespace TECO_INFER
