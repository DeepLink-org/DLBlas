/**
 * PyTorch host implementation for apply_mix operator.
 * Accepts bf16 x + fp32 mix, converts x to fp32 on NPU, runs fp32 kernel,
 * converts result back to bf16.
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/apply_mix_tiling.h"

extern "C" void apply_mix_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                                  uint8_t* x, uint8_t* mix, uint8_t* y, uint8_t* tiling);

namespace ascend_kernel {

at::Tensor apply_mix_torch(const at::Tensor& x, const at::Tensor& mix)
{
    // Input validation
    TORCH_CHECK(x.scalar_type() == at::kBFloat16,
        "apply_mix: x must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(mix.scalar_type() == at::kFloat,
        "apply_mix: mix must be float32, got ", mix.scalar_type());
    TORCH_CHECK(x.is_privateuseone(), "apply_mix: x must be on NPU");
    TORCH_CHECK(mix.is_privateuseone(), "apply_mix: mix must be on NPU");
    TORCH_CHECK(x.dim() == 4, "apply_mix: x must be 4D [n0,n1,mhc,h]");
    TORCH_CHECK(x.size(0) == mix.size(0) && x.size(1) == mix.size(1) && x.size(2) == mix.size(2),
        "apply_mix: x and mix batch dims must match");
    TORCH_CHECK(mix.size(3) == 1, "apply_mix: mix last dim must be 1");

    uint32_t n0 = static_cast<uint32_t>(x.size(0));
    uint32_t n1 = static_cast<uint32_t>(x.size(1));
    uint32_t mhc = static_cast<uint32_t>(x.size(2));
    uint32_t h = static_cast<uint32_t>(x.size(3));

    // Convert bf16 x → fp32 on NPU (framework handles conversion)
    at::Tensor xFp32 = x.to(at::kFloat);
    // mix is already fp32
    at::Tensor mixContig = mix.contiguous();

    // Allocate fp32 output on NPU
    at::Tensor yFp32 = at::empty({n0, n1, h}, xFp32.options());

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0,
        "apply_mix: failed to get NPU core count");

    ApplyMixTilingData tiling = ComputeTiling(n0, n1, mhc, h,
        static_cast<uint32_t>(availableCoreNum));

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(ApplyMixTilingData))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(ApplyMixTilingData),
        &tiling, sizeof(ApplyMixTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    apply_mix_kernel(tiling.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(xFp32.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(mixContig.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(yFp32.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    // Convert fp32 output → bf16
    return yFp32.to(at::kBFloat16);
}

} // namespace ascend_kernel
