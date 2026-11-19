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

#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "interface/include/tecoops.h"

namespace TECO_INFER {

struct plugin_reshape_and_cacheAttrs
    : public tvm::AttrsNode<plugin_reshape_and_cacheAttrs> {
  int64_t num_blocks;
  int64_t block_size;
  TVM_DECLARE_ATTRS(plugin_reshape_and_cacheAttrs,
                     "relay.attrs.plugin_reshape_and_cacheAttrs") {
    TVM_ATTR_FIELD(num_blocks).set_default(0).describe("Number of KV cache blocks");
    TVM_ATTR_FIELD(block_size).set_default(0).describe("Block size (tokens per block)");
  }
};
TVM_REGISTER_NODE_TYPE(plugin_reshape_and_cacheAttrs);

class PluginReshapeAndCacheImpl : public AbstractPluginOp {
 public:
  PluginReshapeAndCacheImpl() = default;

  void InferOutputShape(
      const std::vector<std::vector<int>>& total_input_shape, int n_input,
      std::vector<std::vector<int>>& total_output_shape, int n_output,
      const OpAttr& attr) {
    int num_kv_heads = total_input_shape[0][1];
    int head_size = total_input_shape[0][2];
    int64_t num_blocks = attr.GetAttr<int64_t>("num_blocks");
    int64_t block_size = attr.GetAttr<int64_t>("block_size");

    // 输出: [num_blocks * 2, num_kv_heads, block_size, head_size]
    // 前一半 key_cache, 后一半 value_cache
    total_output_shape[0] = {static_cast<int>(num_blocks * 2), num_kv_heads,
                             static_cast<int>(block_size), head_size};
  }

  void Enqueue(std::shared_ptr<ComputeContext>& ctx) {
    std::cout << "CALL PluginReshapeAndCache enqueue" << std::endl;

    void* key_dev = ctx->GetInputDataPtr("key");
    void* value_dev = ctx->GetInputDataPtr("value");
    void* slot_mapping_dev = ctx->GetInputDataPtr("slot_mapping");

    int64_t num_blocks, block_size;
    ctx->GetAttr("num_blocks", num_blocks);
    ctx->GetAttr("block_size", block_size);

    std::vector<int> key_shape;
    ctx->GetInputShape("key", key_shape);
    int num_tokens = key_shape[0];
    int num_kv_heads = key_shape[1];
    int head_size = key_shape[2];

    void* out_base = ctx->GetOutputDataPtr(0);
    int64_t cache_bytes = num_blocks * block_size * num_kv_heads * head_size * 2;
    void* key_cache_dev = out_base;
    void* value_cache_dev = static_cast<char*>(out_base) + cache_bytes;

    std::cout << "ReshapeAndCache: tokens=" << num_tokens
              << " heads=" << num_kv_heads << " head_size=" << head_size
              << " blocks=" << num_blocks << " block_size=" << block_size << std::endl;

    sdaaStream_t stream = ctx->GetStream();

    tecoopsHandle_t handle;
    tecoopsCreate(&handle);
    tecoopsSetStream(handle, stream);

    // 清零输出
    tecoopsMemset(handle, out_base, 0, cache_bytes * 2);

    tecoopsReshapeAndCache(
        handle,
        key_dev, value_dev,
        static_cast<const int64_t*>(slot_mapping_dev),
        key_cache_dev, value_cache_dev,
        num_tokens, num_kv_heads, head_size,
        static_cast<int>(num_blocks), static_cast<int>(block_size));

    tecoopsDestroy(handle);
  }
};

REGISTER_PLUGIN_OP_IMPL(plugin_reshape_and_cache, PluginReshapeAndCacheImpl)

PLUGIN_REGISTER_OP("plugin_reshape_and_cache")
    .Input("key")
    .Type("Tensor")
    .Desc("Key input with shape [num_tokens, num_kv_heads, head_size]")
    .Input("value")
    .Type("Tensor")
    .Desc("Value input with shape [num_tokens, num_kv_heads, head_size]")
    .Input("slot_mapping")
    .Type("Tensor")
    .Desc("Slot mapping with shape [num_tokens]")
    .AttrType<plugin_reshape_and_cacheAttrs>()
    .Register();

}  // namespace TECO_INFER
