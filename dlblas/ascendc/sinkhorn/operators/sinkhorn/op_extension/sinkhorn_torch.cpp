// Sinkhorn Normalize - PyTorch 接入层
// stream(true) 在返回 ACL stream 前清 queue，防乱序

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/sinkhorn_tiling.h"

extern "C" void sinkhorn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                 uint8_t *x, uint8_t *y, uint8_t *tiling);

namespace ascend_kernel {

at::Tensor sinkhorn_torch(const at::Tensor& x)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat, "sinkhorn_normalize: only FP32 supported");
    TORCH_CHECK(x.is_privateuseone(), "sinkhorn_normalize: input must be on NPU");

    at::Tensor y = at::empty_like(x);

    // x 的 shape 为 [..., batch, 4, 4]
    auto x_sizes = x.sizes();
    TORCH_CHECK(x_sizes.size() >= 3, "sinkhorn_normalize: input must be at least 3D");
    int64_t batch = x_sizes[x_sizes.size() - 3];
    int64_t mhc = x_sizes[x_sizes.size() - 2];
    TORCH_CHECK(mhc == 4, "sinkhorn_normalize: matrix dim must be 4");
    int64_t totalElements = x.numel();
    TORCH_CHECK(totalElements > 0, "sinkhorn_normalize: input must not be empty");

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 查询核数
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // Tiling 计算 (沿 batch 维度多核切分)
    uint32_t usedCoreNum = std::min(static_cast<uint32_t>(availableCoreNum), static_cast<uint32_t>(batch));
    uint32_t rawTileBatch = (static_cast<uint32_t>(batch) + usedCoreNum - 1) / usedCoreNum;
    uint32_t tileBatch = std::min(rawTileBatch, static_cast<uint32_t>(MAX_TILE_MATRICES));
    usedCoreNum = (static_cast<uint32_t>(batch) + tileBatch - 1) / tileBatch;
    uint32_t tailBatch = static_cast<uint32_t>(batch) - (usedCoreNum - 1) * tileBatch;
    if (tailBatch == 0) tailBatch = tileBatch;

    SinkhornTilingData tiling;
    tiling.batch = static_cast<uint32_t>(batch);
    tiling.mhc = static_cast<uint32_t>(mhc);
    tiling.repeat = REPEAT;
    tiling.eps = EPS;
    tiling.tileBatch = tileBatch;
    tiling.tailBatch = tailBatch;
    tiling.usedCoreNum = usedCoreNum;
    tiling.tileElements = 0;

    uint32_t blockNum = usedCoreNum;

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(SinkhornTilingData))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(SinkhornTilingData),
        &tiling, sizeof(SinkhornTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    sinkhorn_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(const_cast<void*>(x.const_data_ptr())),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return y;
}

} // namespace ascend_kernel
