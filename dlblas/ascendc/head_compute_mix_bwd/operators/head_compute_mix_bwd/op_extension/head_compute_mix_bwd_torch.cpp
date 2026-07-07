/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

// Stream sync: stream(true) to clear queue, preventing out-of-order execution
#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/head_compute_mix_bwd_tiling.h"
#include "tiling/sigmoid/sigmoid_tiling.h"

extern "C" void head_compute_mix_bwd_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    uint8_t* input_mix, uint8_t* mhc_scale, uint8_t* mhc_base,
    uint8_t* grad_out, uint8_t* grad_input_mix,
    uint8_t* grad_mhc_scale, uint8_t* grad_mhc_base,
    uint8_t* workspace, uint8_t* tiling);

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor> head_compute_mix_bwd_torch(
    const at::Tensor& input_mix,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    const at::Tensor& grad_out)
{
    TORCH_CHECK(input_mix.scalar_type() == at::kFloat, "only FP32 supported");
    TORCH_CHECK(input_mix.is_privateuseone(), "input_mix must be on NPU");

    auto grad_input_mix = at::empty_like(input_mix);
    auto grad_mhc_scale = at::empty({1}, input_mix.options());
    auto grad_mhc_base = at::empty({4}, input_mix.options());

    // Get dims
    int64_t B = input_mix.size(0);
    int64_t S = input_mix.size(1);
    int64_t C = input_mix.size(2);
    uint32_t total_rows = static_cast<uint32_t>(B * S);
    uint32_t inner_dim = static_cast<uint32_t>(C);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Get core count
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availCores = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availCores);

    // Compute tiling
    HeadComputeMixBwdTiling tiling;
    tiling.total_rows = total_rows;
    tiling.inner_dim = inner_dim;

    uint32_t total_elems = total_rows * inner_dim;
    uint32_t min_bytes_per_core = 4096;
    uint32_t core_num = (total_elems * sizeof(float) + min_bytes_per_core - 1) / min_bytes_per_core;
    if (core_num > static_cast<uint32_t>(availCores)) core_num = static_cast<uint32_t>(availCores);
    if (core_num < 1) core_num = 1;
    tiling.core_num = core_num;
    tiling.rows_per_core = (total_rows + core_num - 1) / core_num;
    tiling.block_num = core_num;
    tiling.tail_rows = total_rows - (core_num - 1) * tiling.rows_per_core;

    // Sigmoid tmp size
    ge::Shape sigmoidShape({static_cast<int64_t>(total_elems)});
    uint32_t sigmoid_tmp_max = 0, sigmoid_tmp_min = 0;
    AscendC::GetSigmoidMaxMinTmpSize(sigmoidShape, sizeof(float), false,
                                     sigmoid_tmp_max, sigmoid_tmp_min);
    tiling.sigmoid_tmp_size = sigmoid_tmp_max;

    // UB split
    const uint32_t ub_size = 196608;
    uint32_t per_row_bytes = inner_dim * sizeof(float);
    uint32_t overhead = 64 + sigmoid_tmp_max + 256 * sizeof(float) + 4 * sizeof(float)
                        + 8 * sizeof(float) + 1 * sizeof(float) + 5 * sizeof(float);
    uint32_t tile_rows = (ub_size - overhead) / (4 * DOUBLE_BUFFER * per_row_bytes);
    if (tile_rows > tiling.rows_per_core) tile_rows = tiling.rows_per_core;
    if (tile_rows < 1) tile_rows = 1;
    tiling.tile_rows = tile_rows;
    tiling.ub_loops = (tiling.rows_per_core + tile_rows - 1) / tile_rows;

    // Workspace
    tiling.ws_offset_stride = ((5 * sizeof(float) + 255) / 256) * 256;
    tiling.workspace_size = core_num * tiling.ws_offset_stride;

    // Tiling → device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(HeadComputeMixBwdTiling))},
        input_mix.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(HeadComputeMixBwdTiling),
        &tiling, sizeof(HeadComputeMixBwdTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // Workspace on device
    at::Tensor wsTensor = at::empty(
        {static_cast<int64_t>(tiling.workspace_size)},
        input_mix.options().dtype(at::kByte));

    // Prepare mhc_base as 8 elements (4 + 4 repeat for broadcast alignment)
    at::Tensor mhc_base_8 = at::empty({8}, mhc_base.options());
    // Copy original mhc_base to first 4 elements, repeat to last 4
    mhc_base_8.narrow(0, 0, 4).copy_(mhc_base);
    mhc_base_8.narrow(0, 4, 4).copy_(mhc_base);

    // Launch kernel
    head_compute_mix_bwd_kernel(
        tiling.block_num, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(input_mix.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(mhc_scale.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(mhc_base_8.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(grad_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(grad_input_mix.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(grad_mhc_scale.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(grad_mhc_base.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(wsTensor.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return std::make_tuple(grad_input_mix, grad_mhc_scale, grad_mhc_base);
}

} // namespace ascend_kernel
