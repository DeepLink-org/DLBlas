/**
 * norm_fn PyTorch 接入层
 *
 * Stream 同步模式：使用 stream(true) 清 queue，确保安全。
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "norm_fn_tiling.h"

// Kernel 入口声明 (来自 norm_fn_kernel.asc)
extern "C" void norm_fn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                               uint8_t *residual, uint8_t *mhc_fn, uint8_t *weight,
                               uint8_t *result, uint8_t *tiling);

namespace ascend_kernel {

at::Tensor norm_fn_torch(
    const at::Tensor& residual,
    const at::Tensor& mhc_fn,
    const c10::optional<at::Tensor>& mhc_norm_weight,
    double mhc_norm_eps)
{
    // 参数校验
    TORCH_CHECK(residual.is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(mhc_fn.is_privateuseone(), "mhc_fn must be on NPU");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");

    bool hasWeight = mhc_norm_weight.has_value() && mhc_norm_weight->defined();
    if (hasWeight) {
        TORCH_CHECK(mhc_norm_weight->is_privateuseone(), "weight must be on NPU");
        TORCH_CHECK(mhc_norm_weight->scalar_type() == at::kFloat, "weight must be float32");
    }

    // 输出: (1, 13, 24) float32
    int64_t mhc_mult = mhc_fn.size(0);  // 24
    at::Tensor result = at::empty({residual.size(0), residual.size(1), mhc_mult},
                                  residual.options().dtype(at::kFloat));

    // stream(true) 清 queue，确保与之前 NPU 操作的正确同步
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Tiling 数据
    NormFnTilingData tiling;
    tiling.total_M     = TOTAL_M;
    tiling.total_N     = TOTAL_N;
    tiling.total_K     = TOTAL_K;
    tiling.tile_K      = TILE_K;
    tiling.tile_K_align = TILE_K_ALIGN;
    tiling.num_K_tiles = NUM_K_TILES;
    tiling.has_weight  = hasWeight;
    tiling.eps         = static_cast<float>(mhc_norm_eps);
    tiling.invK        = 1.0f / static_cast<float>(TOTAL_K);

    uint32_t blockNum = 1;  // 单核

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(NormFnTilingData))},
        residual.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(NormFnTilingData),
        &tiling, sizeof(NormFnTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 准备 weight 指针 (可能为空)
    uint8_t *weightPtr = nullptr;
    if (hasWeight) {
        weightPtr = reinterpret_cast<uint8_t*>(mhc_norm_weight->data_ptr());
    } else {
        // 传一个占位指针，kernel 中不会访问
        weightPtr = reinterpret_cast<uint8_t*>(residual.data_ptr());
    }

    // 调用 kernel
    norm_fn_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(residual.data_ptr()),
        reinterpret_cast<uint8_t*>(mhc_fn.data_ptr()),
        weightPtr,
        reinterpret_cast<uint8_t*>(result.data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.data_ptr()));

    return result;
}

} // namespace ascend_kernel
