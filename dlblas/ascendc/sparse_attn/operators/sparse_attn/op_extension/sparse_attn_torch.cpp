/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * SparseAttn PyTorch 接入层
 *   Stream 同步模式: stream(true) 清 queue
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/sparse_attn_tiling.h"

// Kernel 入口声明
extern "C" void sparse_attn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *q, uint8_t *kv, uint8_t *topk_idxs,
    uint8_t *attn_sink, uint8_t *output, uint8_t *tiling);

namespace ascend_kernel {

at::Tensor sparse_attn_torch(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& attn_sink,
    const at::Tensor& topk_idxs,
    double softmax_scale)
{
    // Type checks
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be float32");
    TORCH_CHECK(topk_idxs.scalar_type() == at::kInt, "topk_idxs must be int32");
    TORCH_CHECK(q.is_privateuseone(), "q must be on NPU");
    TORCH_CHECK(kv.is_privateuseone(), "kv must be on NPU");

    // Extract shapes: q=[b,m,h,d], kv=[b,n,d], idx=[b,m,topk], sink=[h]
    auto qShape = q.sizes();       // [b, m, h, d]
    auto kvShape = kv.sizes();     // [b, n, d]
    auto idxShape = topk_idxs.sizes(); // [b, m, topk]
    auto sinkShape = attn_sink.sizes(); // [h]

    TORCH_CHECK(qShape.size() == 4, "q must be 4D [b, m, h, d]");
    TORCH_CHECK(kvShape.size() == 3, "kv must be 3D [b, n, d]");
    TORCH_CHECK(idxShape.size() == 3, "topk_idxs must be 3D [b, m, topk]");
    TORCH_CHECK(sinkShape.size() == 1, "attn_sink must be 1D [h]");

    int64_t b = qShape[0], m = qShape[1], h = qShape[2], d = qShape[3];
    int64_t n = kvShape[1], topk = idxShape[2];

    TORCH_CHECK(kvShape[0] == b && kvShape[2] == d, "kv shape mismatch");
    TORCH_CHECK(idxShape[0] == b && idxShape[1] == m, "topk_idxs shape mismatch");
    TORCH_CHECK(sinkShape[0] == h, "attn_sink shape mismatch");

    // Output: same as q
    at::Tensor output = at::empty_like(q);

    // Get device info
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t aivNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aivNum);
    TORCH_CHECK(ret == ACL_SUCCESS && aivNum > 0, "failed to get NPU core count");

    // Compute Tiling
    SparseAttnTiling tiling;
    tiling.b    = static_cast<uint32_t>(b);
    tiling.m    = static_cast<uint32_t>(m);
    tiling.n    = static_cast<uint32_t>(n);
    tiling.h    = static_cast<uint32_t>(h);
    tiling.d    = static_cast<uint32_t>(d);
    tiling.topk = static_cast<uint32_t>(topk);

    tiling.totalTasks = static_cast<uint32_t>(b * m);
    tiling.usedCoreNum = static_cast<uint32_t>(std::min<int64_t>(aivNum, b * m));
    if (tiling.usedCoreNum < 1) tiling.usedCoreNum = 1;
    tiling.tasksPerCore = (tiling.totalTasks + tiling.usedCoreNum - 1) / tiling.usedCoreNum;

    uint32_t per_task_ub = 4 * tiling.h * tiling.d + 4 * tiling.h * tiling.topk
                          + 4 * tiling.topk * tiling.d + 8 * tiling.h;
    tiling.tile_m = (UB_AVAIL - 4096) / per_task_ub;
    if (tiling.tile_m < 1) tiling.tile_m = 1;
    if (tiling.tile_m > TILE_M_MAX) tiling.tile_m = TILE_M_MAX;

    tiling.softmax_scale = static_cast<float>(softmax_scale);
    tiling.coreTaskStart = 0;
    tiling.coreTaskCount = 0;

    uint32_t blockNum = tiling.usedCoreNum;

    // stream(true) 清 queue 后返回 aclStream
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Tiling data to device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(SparseAttnTiling))},
        q.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(SparseAttnTiling),
        &tiling, sizeof(SparseAttnTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel
    sparse_attn_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(topk_idxs.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(attn_sink.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(output.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return output;
}

} // namespace ascend_kernel
