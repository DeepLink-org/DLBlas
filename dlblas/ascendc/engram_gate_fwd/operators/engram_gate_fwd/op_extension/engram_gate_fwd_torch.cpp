// engram_gate_fwd PyTorch integration layer
// Stream sync mode: stream(true) to flush queue before kernel launch

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/engram_gate_fwd_tiling.h"

extern "C" void engram_gate_fwd_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *hidden_states, uint8_t *k, uint8_t *v,
    uint8_t *weight_hidden, uint8_t *weight_embed,
    uint8_t *output, uint8_t *raw_dot, uint8_t *gate_score,
    uint8_t *rstd_x, uint8_t *rstd_k, uint8_t *tiling);

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
engram_gate_fwd_torch(
    const at::Tensor& hidden_states,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& weight_hidden,
    const at::Tensor& weight_embed,
    double clamp_value,
    double eps)
{
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16, "hidden_states must be bf16");
    TORCH_CHECK(hidden_states.is_privateuseone(), "hidden_states must be on NPU");

    int64_t num_tokens  = hidden_states.size(0);
    int64_t hc_mult     = hidden_states.size(1);
    int64_t hidden_size = hidden_states.size(2);

    // Output allocation
    auto output     = at::empty_like(hidden_states);
    auto raw_dot    = at::empty({num_tokens, hc_mult},
                                hidden_states.options().dtype(at::kFloat));
    auto gate_score = at::empty({num_tokens, hc_mult},
                                hidden_states.options().dtype(at::kFloat));
    auto rstd_x     = at::empty({num_tokens, hc_mult},
                                hidden_states.options().dtype(at::kFloat));
    auto rstd_k     = at::empty({num_tokens, hc_mult},
                                hidden_states.options().dtype(at::kFloat));

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    EngramGateFwdTilingData tiling;
    ComputeTiling(tiling, static_cast<uint64_t>(num_tokens),
                  static_cast<uint64_t>(hc_mult),
                  static_cast<uint64_t>(hidden_size),
                  static_cast<float>(clamp_value),
                  static_cast<float>(eps),
                  static_cast<uint32_t>(availableCoreNum));

    // UB capacity validation (DAV_2201: 192 KB = 196608 bytes)
    uint64_t ub_usage = ComputeUBUsage(static_cast<uint64_t>(hidden_size),
                                       static_cast<uint64_t>(hc_mult));
    TORCH_CHECK(ub_usage <= UB_CAPACITY_DAV_2201,
        "engram_gate_fwd: hidden_size=", hidden_size, " exceeds UB capacity. ",
        "UB required: ", ub_usage, " bytes > ", UB_CAPACITY_DAV_2201,
        " bytes (192 KB limit). Max safe hidden_size: ~",
        MaxHiddenSizeForUB(static_cast<uint64_t>(hc_mult)));

    uint32_t blockNum = tiling.core_num;

    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(EngramGateFwdTilingData))},
        hidden_states.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(EngramGateFwdTilingData),
        &tiling, sizeof(EngramGateFwdTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    engram_gate_fwd_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(hidden_states.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(k.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(v.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(weight_hidden.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(weight_embed.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(output.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(raw_dot.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(gate_score.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(rstd_x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(rstd_k.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return {output, raw_dot, gate_score, rstd_x, rstd_k};
}

} // namespace ascend_kernel
