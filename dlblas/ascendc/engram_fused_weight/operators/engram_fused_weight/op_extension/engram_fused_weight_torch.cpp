/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Stream sync: stream(true) clears the task queue before returning the ACL stream,
// ensuring correct ordering with prior NPU operations.
//   DONT: stream(false) + direct call -> out-of-order
//   DONT: zeros_like for output -> out-of-order (use empty_like)

#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/engram_fused_weight_tiling.h"

// Kernel entry declaration (from engram_fused_weight_kernel.asc)
extern "C" void engram_fused_weight_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *wh, uint8_t *we, uint8_t *out, uint8_t *tiling);

namespace ascend_kernel {

at::Tensor engram_fused_weight_torch(const at::Tensor& wh_data, const at::Tensor& we_data)
{
    TORCH_CHECK(wh_data.scalar_type() == at::kBFloat16, "wh_data must be bfloat16");
    TORCH_CHECK(we_data.scalar_type() == at::kBFloat16, "we_data must be bfloat16");
    TORCH_CHECK(wh_data.is_privateuseone(), "wh_data must be on NPU");
    TORCH_CHECK(we_data.is_privateuseone(), "we_data must be on NPU");
    TORCH_CHECK(wh_data.sizes() == we_data.sizes(), "wh_data and we_data must have same shape");

    // Per DESIGN.md §1.2: Output is FP32
    at::Tensor output = at::empty_like(wh_data,
        wh_data.options().dtype(at::kFloat));

    int64_t totalElements = wh_data.numel();
    TORCH_CHECK(totalElements > 0, "input tensors must not be empty");

    // stream(true) clears queue before returning ACL stream
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Get available core count
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // Flat tensor: dim0 = numel()
    int64_t hc_mult = 1;
    int64_t hidden_size = totalElements;

    // Compute tiling per DESIGN.md §9.4
    auto tiling = ComputeTiling(hc_mult, hidden_size, static_cast<int32_t>(availableCoreNum));
    uint32_t blockNum = static_cast<uint32_t>(tiling.blockNum);

    // Copy tiling data to NPU via PyTorch tensor (ensures proper sync)
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(EngramFusedWeightTilingData))},
        at::TensorOptions().dtype(at::kByte));
    std::memcpy(tilingTensor.data_ptr(), &tiling, sizeof(EngramFusedWeightTilingData));
    tilingTensor = tilingTensor.to(wh_data.device());

    // Launch kernel
    engram_fused_weight_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(wh_data.data_ptr()),
        reinterpret_cast<uint8_t*>(we_data.data_ptr()),
        reinterpret_cast<uint8_t*>(output.data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.data_ptr()));

    // Synchronize stream to ensure kernel completes before returning.
    // Without this, the output tensor may be used before the kernel finishes.
    aclrtSynchronizeStream(aclStream);

    return output;
}

} // namespace ascend_kernel
