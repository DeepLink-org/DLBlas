/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * sparse_attn_torch.cpp - PyTorch TORCH_LIBRARY adapter for sparse_attn
 *
 * Stream sync: stream(true) mode to clear queue before kernel launch
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/sparse_attn_tiling.h"

extern "C" void sparse_attn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *q, uint8_t *kv, uint8_t *attn_sink, uint8_t *topk_idxs,
    uint8_t *o, uint8_t *tiling);

namespace ascend_kernel {

at::Tensor sparse_attn_torch(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& attn_sink,
    const at::Tensor& topk_idxs,
    double softmax_scale)
{
    // Validation
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be fp32");
    TORCH_CHECK(topk_idxs.scalar_type() == at::kInt, "topk_idxs must be int32");
    TORCH_CHECK(q.is_privateuseone(), "q must be on NPU");
    TORCH_CHECK(kv.is_privateuseone(), "kv must be on NPU");

    uint32_t b = q.size(0);
    uint32_t m = q.size(1);
    uint32_t h = q.size(2);
    uint32_t d = q.size(3);
    uint32_t topk = topk_idxs.size(2);
    uint32_t n = kv.size(1);

    at::Tensor o = at::empty_like(q);

    // Get NPU stream (stream(true) clears queue)
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Get device info
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t aicCoreNum = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aicCoreNum);
    int64_t ubSize = 192 * 1024; // DAV_2201 UB size

    // Compute tiling
    SparseAttnTilingData tiling = ComputeSparseAttnTiling(
        b, m, h, d, topk, n, (float)softmax_scale,
        (uint32_t)ubSize, (uint32_t)aicCoreNum);

    uint32_t blockNum = tiling.usedCoreNum;

    // Copy tiling to device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(SparseAttnTilingData))},
        q.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(SparseAttnTilingData),
        &tiling, sizeof(SparseAttnTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel
    sparse_attn_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(attn_sink.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(topk_idxs.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(o.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return o;
}

} // namespace ascend_kernel
