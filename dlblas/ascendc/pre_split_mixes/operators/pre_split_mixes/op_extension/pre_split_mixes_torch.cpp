/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/pre_split_mixes_tiling.h"

// Kernel 入口声明 (输出指针通过 tiling struct 传递)
extern "C" void pre_split_mixes_kernel(uint32_t blockDim, void *l2Ctrl,
    aclrtStream stream,
    uint8_t *input_mixes, uint8_t *mhc_scale, uint8_t *mhc_base,
    uint8_t *tiling);

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor> pre_split_mixes_torch(
    const at::Tensor& input_mixes,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    int64_t mhc_mult,
    double mhc_pre_eps,
    double mhc_post_mult_value)
{
    TORCH_CHECK(input_mixes.scalar_type() == at::kFloat, "input_mixes must be FP32");
    TORCH_CHECK(mhc_scale.scalar_type() == at::kFloat, "mhc_scale must be FP32");
    TORCH_CHECK(mhc_base.scalar_type() == at::kFloat, "mhc_base must be FP32");
    TORCH_CHECK(input_mixes.is_privateuseone(), "input_mixes must be on NPU");

    auto batch = input_mixes.size(0);
    auto seq_len = input_mixes.size(1);
    int64_t totalRows = batch * seq_len;
    int32_t m = static_cast<int32_t>(mhc_mult);
    int64_t M3 = 2LL * m + (int64_t)m * m;

    TORCH_CHECK(input_mixes.size(2) == M3, "input_mixes dim2 must equal M3");
    TORCH_CHECK(totalRows > 0, "input must not be empty");

    // 输出 tensor
    at::Tensor pre_mix  = at::empty({batch, seq_len, m}, input_mixes.options());
    at::Tensor post_mix = at::empty({batch, seq_len, m}, input_mixes.options());
    at::Tensor comb_mix = at::empty({batch, seq_len, m * m}, input_mixes.options());

    // stream(true) 清 queue
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 查询可用核数
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM,
        &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0,
        "failed to get NPU core count");

    // === Tiling 计算 ===
    PreSplitMixesTilingData tiling;
    tiling.totalRows = totalRows;
    tiling.mhcMult   = m;
    tiling.mhcMult3  = M3;
    tiling.mhcPreEps = static_cast<float>(mhc_pre_eps);
    tiling.mhcPostMultValue = static_cast<float>(mhc_post_mult_value);

    // 多核切分
    int64_t totalElems = totalRows * M3;
    int64_t coreNumByElems = (totalElems * 32 + 32767) / 32768;
    if (coreNumByElems < 1) coreNumByElems = 1;
    int32_t coreNum = static_cast<int32_t>(
        std::min(coreNumByElems, availableCoreNum));
    tiling.coreNum = coreNum;

    int64_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;
    tiling.rowsPerCore = rowsPerCore;
    tiling.tailRows = totalRows - (coreNum - 1) * rowsPerCore;

    // Sigmoid temp
    tiling.sigmoidTmpBufSize = 8192;  // 8KB conservative
    tiling.rowsPerChunk = 1;
    tiling.ubLoopPerCore = rowsPerCore;
    tiling.ubLoopTailCore = tiling.tailRows;

    uint32_t blockNum = static_cast<uint32_t>(tiling.coreNum);

    // 输出指针写入 tiling struct (减少 kernel 参数个数)
    tiling.preGmAddr  = reinterpret_cast<uint64_t>(pre_mix.data_ptr());
    tiling.postGmAddr = reinterpret_cast<uint64_t>(post_mix.data_ptr());
    tiling.combGmAddr = reinterpret_cast<uint64_t>(comb_mix.data_ptr());

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(PreSplitMixesTilingData))},
        input_mixes.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(),
        sizeof(PreSplitMixesTilingData),
        &tiling, sizeof(PreSplitMixesTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE);

    // Kernel 调用 (4 参数: input, scale, bias, tiling)
    pre_split_mixes_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(input_mixes.data_ptr()),
        reinterpret_cast<uint8_t*>(mhc_scale.data_ptr()),
        reinterpret_cast<uint8_t*>(mhc_base.data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.data_ptr()));

    return std::make_tuple(pre_mix, post_mix, comb_mix);
}

} // namespace ascend_kernel
