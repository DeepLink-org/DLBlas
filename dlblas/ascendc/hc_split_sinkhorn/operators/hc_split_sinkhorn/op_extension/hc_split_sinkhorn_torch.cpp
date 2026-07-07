/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/hc_split_sinkhorn_tiling.h"

extern "C" void hc_split_sinkhorn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *mixes, uint8_t *pre, uint8_t *post, uint8_t *comb, uint8_t *tiling);

namespace ascend_kernel {

void hc_split_sinkhorn_torch(
    const at::Tensor& mixes,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& pre,
    at::Tensor& post,
    at::Tensor& comb)
{
    TORCH_CHECK(mixes.scalar_type() == at::kFloat, "mixes must be FP32");
    TORCH_CHECK(mixes.is_privateuseone(), "mixes must be on NPU");

    auto b = mixes.size(0);
    auto s = mixes.size(1);
    uint64_t hc = static_cast<uint64_t>(hc_mult);
    uint64_t totalBatch = static_cast<uint64_t>(b) * static_cast<uint64_t>(s);
    uint32_t iters = static_cast<uint32_t>(sinkhorn_iters);
    float epsVal = static_cast<float>(eps);

    at::Tensor hcScaleCpu = hc_scale.cpu().contiguous();
    at::Tensor hcBaseCpu = hc_base.cpu().contiguous();
    const float* scalePtr = hcScaleCpu.data_ptr<float>();
    const float* basePtr = hcBaseCpu.data_ptr<float>();

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    HcSplitSinkhornTiling tiling;
    float hcScaleArr[3], hcBaseArr[HC_SPLIT_SINKHORN_MAX_MIX_HC];
    for (int i = 0; i < 3; i++) hcScaleArr[i] = scalePtr[i];
    uint64_t mixHc = (2 + hc) * hc;
    for (uint64_t i = 0; i < mixHc && i < HC_SPLIT_SINKHORN_MAX_MIX_HC; i++) hcBaseArr[i] = basePtr[i];
    for (uint64_t i = mixHc; i < HC_SPLIT_SINKHORN_MAX_MIX_HC; i++) hcBaseArr[i] = 0.0f;

    ComputeTiling(tiling, totalBatch, hc, iters, epsVal, availableCoreNum, hcScaleArr, hcBaseArr);

    uint32_t blockNum = (availableCoreNum < static_cast<int64_t>(tiling.usedCoreNum))
        ? static_cast<uint32_t>(availableCoreNum) : tiling.usedCoreNum;

    // Tiling 数据搬到 device (使用 aclrtMalloc 确保生命周期跨越 kernel 执行)
    uint8_t *tilingDevice = nullptr;
    aclrtMalloc((void **)&tilingDevice, sizeof(HcSplitSinkhornTiling), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingDevice, sizeof(HcSplitSinkhornTiling),
        &tiling, sizeof(HcSplitSinkhornTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    hc_split_sinkhorn_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(mixes.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(pre.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(post.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(comb.mutable_data_ptr()),
        tilingDevice);

    aclrtSynchronizeStream(aclStream);
    aclrtFree(tilingDevice);
}

} // namespace ascend_kernel
