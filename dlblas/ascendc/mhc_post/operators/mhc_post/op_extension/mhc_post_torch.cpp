/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

// mhc_post PyTorch extension - host implementation
// Uses stream(true) + tensor-based tiling pattern for correct ordering.

// ACL headers must come before torch headers to avoid macro conflicts
#include "acl/acl.h"

#include <torch/torch.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "../op_kernel/mhc_post_tiling.h"
#include "ops.h"

// Kernel declaration: matches ASC-generated entry point
// From mhc_post_kernel.asc:
//   extern "C" __global__ __vector__ void mhc_post_kernel(
//       GM_ADDR x, GM_ADDR residual, GM_ADDR post_layer_mix,
//       GM_ADDR comb_res_mix, GM_ADDR output, GM_ADDR tiling)
// ASC wraps this with calling convention: (blockDim, l2ctrl, stream, kernel_args...)
extern "C" void mhc_post_kernel(
    uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    uint8_t* x, uint8_t* residual, uint8_t* post_layer_mix,
    uint8_t* comb_res_mix, uint8_t* output, uint8_t* tiling);

namespace ascend_kernel {

static void ComputeTiling(uint32_t n0, uint32_t n1, uint32_t h,
                          uint32_t blockNum, MhcPostTiling& tiling)
{
    tiling.n0       = n0;
    tiling.n1       = n1;
    tiling.blockNum = blockNum;
    tiling.bTile    = B_TILE;
    tiling.cTile    = C_TILE;
    tiling.h        = h;
}

at::Tensor mhc_post_torch(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix)
{
    // Validate inputs
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(residual.is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(post_layer_mix.is_privateuseone(), "post_layer_mix must be on NPU");
    TORCH_CHECK(comb_res_mix.is_privateuseone(), "comb_res_mix must be on NPU");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (n0, n1, h)");
    TORCH_CHECK(residual.dim() == 4, "residual must be 4D (n0, n1, M, h)");

    uint32_t n0 = static_cast<uint32_t>(x.size(0));
    uint32_t n1 = static_cast<uint32_t>(x.size(1));
    uint32_t h  = static_cast<uint32_t>(x.size(2));

    // Get current device and determine blockNum
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    uint32_t blockNum = static_cast<uint32_t>(
        std::min({availableCoreNum,
                  static_cast<int64_t>(MAX_CORE_NUM),
                  static_cast<int64_t>(n1)}));
    if (blockNum < 1) blockNum = 1;

    // Compute tiling
    MhcPostTiling tilingHost;
    ComputeTiling(n0, n1, h, blockNum, tilingHost);

    // Allocate tiling as a tensor (kept alive by PyTorch ref-counting through kernel completion)
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(MhcPostTiling))},
        x.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(MhcPostTiling),
                &tilingHost, sizeof(MhcPostTiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate output: (n0, n1, M, h) bf16
    at::Tensor output = at::empty(
        {static_cast<int64_t>(n0),
         static_cast<int64_t>(n1),
         static_cast<int64_t>(MHC_MULT),
         static_cast<int64_t>(h)},
        x.options());

    // stream(true): clear queue before returning ACL stream to prevent ordering issues
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // Launch kernel
    mhc_post_kernel(
        blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.data_ptr()),
        reinterpret_cast<uint8_t*>(residual.data_ptr()),
        reinterpret_cast<uint8_t*>(post_layer_mix.data_ptr()),
        reinterpret_cast<uint8_t*>(comb_res_mix.data_ptr()),
        reinterpret_cast<uint8_t*>(output.data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.data_ptr()));

    return output;
}

}  // namespace ascend_kernel
