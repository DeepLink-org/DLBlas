/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/act_quant_kernel_tiling.h"

// Kernel entry declaration (from act_quant_kernel_kernel.asc)
extern "C" void act_quant_kernel_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *x, uint8_t *q, uint8_t *s, uint8_t *tiling);

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> act_quant_kernel_torch(
    const at::Tensor& x, int64_t group_size, double eps, bool scale_ue8m0)
{
    // Validate input
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");

    auto dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
        "x dtype must be bf16 or fp16, got ", dtype);

    int64_t N = x.size(-1);
    int64_t numel = x.numel();
    TORCH_CHECK(numel > 0, "x must not be empty");
    TORCH_CHECK(N % group_size == 0,
        "last dimension N=", N, " must be divisible by group_size=", group_size);
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    uint32_t dsize = (dtype == at::kBFloat16) ? 2 : 2;  // both bf16 and fp16 = 2 bytes

    // Compute output shapes
    auto sShape = x.sizes().vec();
    sShape.back() = N / group_size;

    at::Tensor x_q = at::empty_like(x, x.options().dtype(at::kByte));  // fp8 stored as uint8
    at::Tensor x_s = at::empty(sShape, x.options().dtype(at::kFloat));

    // Stream
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Device and core count
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // Compute tiling
    uint32_t totalElements = static_cast<uint32_t>(numel);
    uint32_t groupSize = static_cast<uint32_t>(group_size);

    ActQuantTiling tiling;
    tiling.numGroups = totalElements / groupSize;
    tiling.groupSize = groupSize;
    tiling.groupSizeAlign = calcGroupSizeAlign(groupSize, dsize);
    tiling.fp8Max = FP8_E4M3FN_MAX;
    tiling.fp8Min = FP8_E4M3FN_MIN;
    tiling.eps = static_cast<float>(eps);
    tiling.scaleUe8m0 = scale_ue8m0;
    tiling.inputDtype = (dtype == at::kBFloat16) ? InputDtype::BF16 : InputDtype::FP16;

    uint32_t numGroups = tiling.numGroups;
    tiling.coreGroups = (numGroups + availableCoreNum - 1) / availableCoreNum;
    if (tiling.coreGroups < 1) tiling.coreGroups = 1;

    // tileGroups based on group size
    if (groupSize >= 128) {
        tiling.tileGroups = 128;
    } else if (groupSize >= 64) {
        tiling.tileGroups = 256;
    } else if (groupSize >= 32) {
        tiling.tileGroups = 512;
    } else {
        tiling.tileGroups = 1024;
    }
    if (tiling.tileGroups > tiling.coreGroups) tiling.tileGroups = tiling.coreGroups;
    if (tiling.tileGroups < 1) tiling.tileGroups = 1;

    uint32_t blockNum = (numGroups + tiling.coreGroups - 1) / tiling.coreGroups;
    if (blockNum < 1) blockNum = 1;
    if (blockNum > (uint32_t)availableCoreNum) blockNum = (uint32_t)availableCoreNum;

    // Copy tiling to device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(ActQuantTiling))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(ActQuantTiling),
        &tiling, sizeof(ActQuantTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel
    act_quant_kernel_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.data_ptr()),
        reinterpret_cast<uint8_t*>(x_q.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(x_s.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return std::make_tuple(x_q, x_s);
}

} // namespace ascend_kernel
