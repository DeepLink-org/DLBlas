/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

// ============================================================================
// PyTorch 接入层 - Expand 算子
// ============================================================================
//
// Stream 同步:
//   stream(true) 在返回 ACL stream 前清 queue，确保与之前 NPU 操作的正确同步
//   禁止使用 stream(false)：不清 queue + 直接调用 kernel = 乱序风险
// ============================================================================

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/expand_kenel_fwd_tiling.h"

// extern 声明 kernel 入口
extern "C" void expand_kenel_fwd_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                                        uint8_t* x, uint8_t* y, uint8_t* tiling);

namespace ascend_kernel {

static inline int64_t CeilDiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static inline int64_t AlignUp16(int64_t x) {
    return ((x + 15) / 16) * 16;
}

at::Tensor expand_kenel_fwd_torch(const at::Tensor& x, int64_t mhc_mult)
{
    TORCH_CHECK(mhc_mult > 0, "mhc_mult must be positive");
    TORCH_CHECK(x.dim() >= 2, "input tensor must have at least 2 dims");
    TORCH_CHECK(x.is_privateuseone(), "input must be on NPU");

    // 计算形状
    // 输入: (..., H) → 展平为 (N, H) where N = product of all dims except last
    // 输出: (..., M, H)
    int64_t H = x.size(-1);

    // H 对齐校验 (M1/M2 fix): H 必须是 16 的倍数 (32B 对齐要求)
    TORCH_CHECK(H % 16 == 0,
        "expand_kenel_fwd: H must be a multiple of 16 (32-byte alignment requirement). "
        "Got H=", H, ". Common LLM hidden sizes (768, 1280, 2048, 4096, etc.) are all compatible.");

    int64_t totalRows = 1;
    for (int64_t i = 0; i < x.dim() - 1; i++) {
        totalRows *= x.size(i);
    }
    int64_t M = mhc_mult;
    int64_t totalElements = totalRows * H;
    TORCH_CHECK(totalElements > 0, "input tensor must not be empty");

    // 构造输出形状
    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end() - 1);
    outShape.push_back(M);
    outShape.push_back(H);

    // 输出 tensor (空的可写)
    at::Tensor y = at::empty(outShape, x.options());

    // stream(true) 在返回 ACL stream 前清 queue
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 查询核数
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // 确定 dtypeSize
    size_t dtypeSize = 0;
    if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
        dtypeSize = 2;
    } else if (x.scalar_type() == at::kFloat) {
        dtypeSize = 4;
    } else {
        TORCH_CHECK(false, "unsupported dtype: ", x.scalar_type());
    }

    // ==========================================================================
    // Tiling 计算
    // ==========================================================================
    // 计算 tileH: UB buffer 需要对齐到 16 倍数 (32B)
    int64_t paddedH = AlignUp16(H);
    int64_t maxTileH = static_cast<int64_t>(UB_BUDGET_BYTES) / ((M + 2) * static_cast<int64_t>(dtypeSize));
    maxTileH = (maxTileH / 16) * 16;

    int64_t tileH;
    int64_t tilesPerRow;
    int64_t tailH;
    if (paddedH <= maxTileH) {
        tileH = paddedH;
        tilesPerRow = 1;
        tailH = H;
    } else {
        tileH = (maxTileH / 16) * 16;
        if (tileH < 16) tileH = 16;
        tilesPerRow = CeilDiv(H, tileH);
        tailH = H - (tilesPerRow - 1) * tileH;
    }

    int64_t usedCoreCnt = 1;
    if (availableCoreNum > 0 && totalRows > 0) {
        usedCoreCnt = std::min(availableCoreNum, totalRows);
    }
    int64_t rowsPerCore = CeilDiv(totalRows, usedCoreCnt);
    int64_t totalTiles = totalRows * tilesPerRow;
    uint32_t blockNum = static_cast<uint32_t>(usedCoreCnt);

    // 填充 Tiling 结构体
    ExpandTilingData tiling;
    tiling.totalRows = totalRows;
    tiling.H = H;
    tiling.M = M;
    tiling.tileH = tileH;
    tiling.rowsPerCore = rowsPerCore;
    tiling.usedCoreCnt = usedCoreCnt;
    tiling.totalTiles = totalTiles;
    tiling.tailH = tailH;
    tiling.dtypeSize = static_cast<uint32_t>(dtypeSize);

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(ExpandTilingData))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(ExpandTilingData),
                &tiling, sizeof(ExpandTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 启动 kernel
    expand_kenel_fwd_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return y;
}

} // namespace ascend_kernel
