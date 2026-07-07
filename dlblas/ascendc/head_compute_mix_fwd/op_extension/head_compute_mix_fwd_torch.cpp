/**
 * @file head_compute_mix_fwd_torch.cpp
 * @brief PyTorch extension layer for head_compute_mix_fwd operator.
 *
 * Stream sync: stream(true) clears queue before kernel call (safe pattern).
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/head_compute_mix_fwd_tiling.h"

extern "C" void head_compute_mix_fwd_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *input, uint8_t *output, uint8_t *tiling);

namespace ascend_kernel {

/**
 * @brief Compute tiling parameters for PyTorch extension.
 */
static void ComputeTiling(HeadComputeMixFwdTilingData* tiling,
                          int64_t dim0, int64_t availableCoreNum)
{
    tiling->dim0 = dim0;

    int64_t coreNum = (dim0 * 16 + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    if (coreNum > availableCoreNum) coreNum = availableCoreNum;
    if (coreNum < 1) coreNum = 1;
    tiling->coreNum = static_cast<int32_t>(coreNum);

    int64_t blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1)
                          / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    tiling->blockFormer = blockFormer;

    int64_t blockNum = (dim0 + blockFormer - 1) / blockFormer;
    tiling->blockNum = blockNum;
    tiling->blockTail = dim0 - (blockNum - 1) * blockFormer;

    tiling->ubFormer = UB_FORMER_HALF;

    int64_t ubLoopF = (blockFormer + tiling->ubFormer - 1) / tiling->ubFormer;
    tiling->ubLoopOfFormerBlock = ubLoopF;
    tiling->ubTailOfFormerBlock = blockFormer - (ubLoopF - 1) * tiling->ubFormer;

    if (tiling->blockTail > 0) {
        int64_t ubLoopT = (tiling->blockTail + tiling->ubFormer - 1) / tiling->ubFormer;
        tiling->ubLoopOfTailBlock = ubLoopT;
        tiling->ubTailOfTailBlock = tiling->blockTail - (ubLoopT - 1) * tiling->ubFormer;
    } else {
        tiling->ubLoopOfTailBlock = tiling->ubLoopOfFormerBlock;
        tiling->ubTailOfTailBlock = tiling->ubTailOfFormerBlock;
    }
}

at::Tensor head_compute_mix_fwd_torch(
    const at::Tensor& input_mix,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    double mhc_pre_eps)
{
    TORCH_CHECK(input_mix.scalar_type() == at::kHalf, "input_mix must be FP16");
    TORCH_CHECK(input_mix.is_privateuseone(), "input_mix must be on NPU");

    // Flatten to 1D
    auto input_flat = input_mix.reshape({-1});
    int64_t dim0 = input_flat.numel();
    TORCH_CHECK(dim0 > 0, "input tensor must not be empty");

    at::Tensor output = at::empty_like(input_flat);

    // stream(true) clears queue before kernel call
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Get core count
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // Compute tiling
    HeadComputeMixFwdTilingData tiling;
    ComputeTiling(&tiling, dim0, availableCoreNum);

    // Fill scalar parameters
    at::Tensor scaleCpu = mhc_scale.cpu();
    at::Tensor baseCpu = mhc_base.cpu();
    tiling.mhc_scale_f32 = static_cast<float>(scaleCpu.item<at::Half>());
    tiling.mhc_pre_eps_f32 = static_cast<float>(mhc_pre_eps);
    auto baseAcc = baseCpu.accessor<at::Half, 1>();
    for (int i = 0; i < 4; i++) {
        tiling.mhc_base_f32[i] = static_cast<float>(baseAcc[i]);
    }

    // Copy tiling to device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(HeadComputeMixFwdTilingData))},
        input_flat.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(HeadComputeMixFwdTilingData),
        &tiling, sizeof(HeadComputeMixFwdTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel
    head_compute_mix_fwd_kernel(
        static_cast<uint32_t>(tiling.blockNum), nullptr, aclStream,
        reinterpret_cast<uint8_t*>(input_flat.data_ptr()),
        reinterpret_cast<uint8_t*>(output.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return output.reshape(input_mix.sizes());
}

} // namespace ascend_kernel
