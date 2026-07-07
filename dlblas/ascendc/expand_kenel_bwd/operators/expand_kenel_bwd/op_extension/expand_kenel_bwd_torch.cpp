/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

// ============================================================================
// PyTorch 接入层 - Expand Kernel Backward 算子
// ============================================================================

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/expand_kenel_bwd_tiling.h"

extern "C" void expand_kenel_bwd_kernel(uint32_t blockDim, void* l2Ctrl,
                                        aclrtStream stream,
                                        uint8_t* oGrad, uint8_t* out,
                                        uint8_t* workspace, uint8_t* tiling);

namespace ascend_kernel {

static void ComputeTiling(ExpandKernelBwdTilingData& tiling,
                          uint64_t A1, uint64_t R, uint64_t A0,
                          uint64_t blockDim)
{
    uint64_t tileA0Len = ((A0 + A0_TILE_BASE - 1) / A0_TILE_BASE) * A0_TILE_BASE;
    uint64_t a0Outer = (A0 + tileA0Len - 1) / tileA0Len;
    uint64_t totalTiles = A1 * a0Outer;
    uint64_t tilesPerCore = (totalTiles + blockDim - 1) / blockDim;
    uint64_t usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    uint64_t tailCoreTiles = totalTiles % tilesPerCore;
    if (tailCoreTiles == 0 && totalTiles > 0) {
        tailCoreTiles = tilesPerCore;
    }

    tiling.A1            = A1;
    tiling.R             = R;
    tiling.A0            = A0;
    tiling.tileA0Len     = tileA0Len;
    tiling.a0Outer       = a0Outer;
    tiling.totalTiles    = totalTiles;
    tiling.tilesPerCore  = tilesPerCore;
    tiling.tailCoreTiles = tailCoreTiles;
    tiling.usedCoreNum   = usedCoreNum;
    // half = 2 bytes
    tiling.inputSize     = static_cast<uint32_t>(A1 * R * A0 * 2);
    tiling.outputSize    = static_cast<uint32_t>(A1 * A0 * 2);
}

at::Tensor expand_kenel_bwd_torch(const at::Tensor& o_grad)
{
    TORCH_CHECK(o_grad.scalar_type() == at::kHalf, "only FP16 supported");
    TORCH_CHECK(o_grad.is_privateuseone(), "o_grad must be on NPU");
    TORCH_CHECK(o_grad.dim() == 4, "o_grad must be 4D: (n0, n1, mhc_mult, h)");

    int64_t n0       = o_grad.size(0);
    int64_t n1       = o_grad.size(1);
    int64_t mhc_mult = o_grad.size(2);
    int64_t h        = o_grad.size(3);

    // 输出 shape: (n0, n1, h)
    at::Tensor out = at::empty({n0, n1, h}, o_grad.options());

    // 合轴
    uint64_t A1 = static_cast<uint64_t>(n0 * n1);
    uint64_t R  = static_cast<uint64_t>(mhc_mult);
    uint64_t A0 = static_cast<uint64_t>(h);

    // 获取 stream (清 queue 模式)
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 获取核数
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                   &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0,
                "failed to get NPU core count");

    // Tiling 计算
    uint64_t blockDim = static_cast<uint64_t>(availableCoreNum);
    ExpandKernelBwdTilingData tiling;
    ComputeTiling(tiling, A1, R, A0, blockDim);

    uint32_t blockNum = static_cast<uint32_t>(tiling.usedCoreNum);

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(ExpandKernelBwdTilingData))},
        o_grad.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(ExpandKernelBwdTilingData),
                &tiling, sizeof(ExpandKernelBwdTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // workspace: 本算子无需额外 workspace
    uint8_t* workspace = nullptr;

    expand_kenel_bwd_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(o_grad.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(out.mutable_data_ptr()),
        workspace,
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return out;
}

} // namespace ascend_kernel
