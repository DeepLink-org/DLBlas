// engram_gate_w_reduce PyTorch 接入层
// 注意事项见 SKILL.md Step 2
//
// Stream 同步模式: stream(true) + 函数调用（清 queue，安全）

#include <cstdint>
#include "acl/acl.h"                        // 必须在 torch 之前 include
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/engram_gate_w_reduce_tiling.h"

// ---- extern 声明 kernel 入口（不 include .asc！）----
extern "C" void engram_gate_w_reduce_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *gm1, uint8_t *gm2, uint8_t *gm3, uint8_t *gm4, uint8_t *gm5,
    uint8_t *tiling);

// ---- Host 侧 Tiling 计算 ----
static EngramGateWReduceTiling ComputeTiling(uint32_t hiddenSize, int64_t availableCoreNum)
{
    uint32_t tileHiddenLen = (hiddenSize + static_cast<uint32_t>(availableCoreNum) - 1)
                             / static_cast<uint32_t>(availableCoreNum);
    if (tileHiddenLen < 1) tileHiddenLen = 1;
    uint32_t blockNum = (hiddenSize + tileHiddenLen - 1) / tileHiddenLen;
    uint32_t tileA0Len = tileHiddenLen * N_CHANNELS;
    uint32_t tailHiddenLen = hiddenSize - (blockNum - 1) * tileHiddenLen;
    uint32_t tailA0Len = tailHiddenLen * N_CHANNELS;

    EngramGateWReduceTiling tiling;
    tiling.blockDim = blockNum;
    tiling.hiddenSize = hiddenSize;
    tiling.tileHiddenLen = tileHiddenLen;
    tiling.tileA0Len = tileA0Len;
    tiling.tailHiddenLen = tailHiddenLen;
    tiling.tailA0Len = tailA0Len;
    tiling.R = R_DIM;
    return tiling;
}

// ---- 算子实现 ----
namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> engram_gate_w_reduce_torch(
    const at::Tensor& grad_w_partial,
    const at::Tensor& weight_hidden,
    const at::Tensor& weight_embed,
    const at::Tensor& grad_weight_hidden,
    const at::Tensor& grad_weight_embed)
{
    // 1. 参数校验
    TORCH_CHECK(grad_w_partial.scalar_type() == at::kFloat, "grad_w_partial must be FP32");
    TORCH_CHECK(weight_hidden.scalar_type() == at::kBFloat16, "weight_hidden must be BF16");
    TORCH_CHECK(weight_embed.scalar_type() == at::kBFloat16, "weight_embed must be BF16");
    TORCH_CHECK(grad_weight_hidden.scalar_type() == at::kFloat, "grad_weight_hidden must be FP32");
    TORCH_CHECK(grad_weight_embed.scalar_type() == at::kFloat, "grad_weight_embed must be FP32");
    TORCH_CHECK(grad_w_partial.is_privateuseone(), "grad_w_partial must be on NPU");
    TORCH_CHECK(grad_weight_hidden.is_privateuseone(), "grad_weight_hidden must be on NPU");
    TORCH_CHECK(grad_weight_embed.is_privateuseone(), "grad_weight_embed must be on NPU");

    // Shape 检查
    // grad_w_partial: [108, 4, H]
    // weight_* / grad_weight_*: [4, H]
    TORCH_CHECK(grad_w_partial.dim() == 3, "grad_w_partial must be 3D [108, 4, H]");
    int64_t R = grad_w_partial.size(0);     // 108
    int64_t C = grad_w_partial.size(1);     // 4
    int64_t H = grad_w_partial.size(2);     // hidden_size
    TORCH_CHECK(R == 108 && C == 4, "grad_w_partial shape must be [108, 4, H]");
    TORCH_CHECK(weight_hidden.dim() == 2 && weight_hidden.size(0) == 4
                && weight_hidden.size(1) == H, "weight_hidden shape must be [4, H]");
    TORCH_CHECK(weight_embed.dim() == 2 && weight_embed.size(0) == 4
                && weight_embed.size(1) == H, "weight_embed shape must be [4, H]");
    TORCH_CHECK(grad_weight_hidden.sizes() == weight_hidden.sizes(),
                "grad_weight_hidden shape must be [4, H]");
    TORCH_CHECK(grad_weight_embed.sizes() == weight_hidden.sizes(),
                "grad_weight_embed shape must be [4, H]");

    // 2. 分配输出（clone 以保持 in-place 语义安全）
    at::Tensor y_hidden = grad_weight_hidden.clone();
    at::Tensor y_embed = grad_weight_embed.clone();

    // 3. 获取框架 stream（stream(true) 清 queue）
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 4. 查询可用核数 & 计算 Tiling
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    EngramGateWReduceTiling tiling = ComputeTiling(static_cast<uint32_t>(H), availableCoreNum);

    // 5. Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(EngramGateWReduceTiling))},
        grad_w_partial.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(EngramGateWReduceTiling),
        &tiling, sizeof(EngramGateWReduceTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // 6. 调用 kernel
    engram_gate_w_reduce_kernel(
        tiling.blockDim, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(grad_w_partial.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(weight_hidden.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(weight_embed.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y_hidden.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y_embed.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return {y_hidden, y_embed};
}

} // namespace ascend_kernel
