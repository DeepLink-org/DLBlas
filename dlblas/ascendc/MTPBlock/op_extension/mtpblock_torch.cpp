// MTPBlock PyTorch Extension - mtpblock_torch.cpp
// K4 hc_post implementation

#include <cstdint>
#include "acl/acl.h"  // must be before torch headers
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "ops.h"
#include "../op_kernel/mtpblock_tiling.h"

// ASC host stub signature: the bisheng compiler generates a wrapper that expects
// (blockDim, l2Ctrl, stream, ...kernel_args...) — the first three params are
// injected by the compiler, not present in the original AscendC kernel signature.
extern "C" void k4_hc_post_kernel(
    uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    void* x, void* residual, void* post, void* comb,
    void* out, void* tiling);

namespace ascend_kernel {

at::Tensor mtpblock_hc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    // Parameter validation
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(residual.is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(post.is_privateuseone(), "post must be on NPU");
    TORCH_CHECK(comb.is_privateuseone(), "comb must be on NPU");

    auto b = x.size(0);
    auto s = x.size(1);
    auto d = x.size(2);
    auto hc = residual.size(2);

    // Allocate output — use empty_like (NOT zeros_like, which would enqueue
    // an NPU op outside our stream control)
    auto out = at::empty_like(residual);

    // Get framework stream with queue flush
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Query available cores
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0,
        "failed to get NPU core count");

    // Dynamic usedCoreNum: min(total_tokens, available_cores)
    uint32_t totalTokens = static_cast<uint32_t>(b * s);
    uint32_t usedCoreNum = (totalTokens < static_cast<uint32_t>(availableCoreNum))
        ? totalTokens : static_cast<uint32_t>(availableCoreNum);
    if (usedCoreNum == 0) usedCoreNum = 1;

    // Build Tiling
    K4HcPostTiling tiling = {};
    tiling.base.kernelType = MTP_K4_HC_POST;
    tiling.base.b = static_cast<uint32_t>(b);
    tiling.base.s = static_cast<uint32_t>(s);
    tiling.base.hc = static_cast<uint32_t>(hc);
    tiling.base.d = static_cast<uint32_t>(d);
    tiling.base.tile_s = (totalTokens < 8) ? totalTokens : 8;
    tiling.base.usedCoreNum = usedCoreNum;
    tiling.base.blockNum = usedCoreNum;

    // Copy tiling to device via a torch tensor
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(K4HcPostTiling))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(K4HcPostTiling),
        &tiling, sizeof(K4HcPostTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // Launch kernel: (blockDim, l2Ctrl, stream, ...kernel_args...)
    k4_hc_post_kernel(
        usedCoreNum, nullptr, aclStream,
        x.data_ptr(), residual.data_ptr(), post.data_ptr(),
        comb.data_ptr(), out.data_ptr(),
        tilingTensor.mutable_data_ptr());

    return out;
}

} // namespace ascend_kernel
