/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// Register operator signature: engram_fused_weight(BF16, BF16) -> FP32
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("engram_fused_weight(Tensor wh_data, Tensor we_data) -> Tensor");
}

// Bind NPU implementation
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("engram_fused_weight", TORCH_FN(ascend_kernel::engram_fused_weight_torch));
}

// Meta backend registration (required for torch.compile / fx)
// Per DESIGN.md §1.2: Output is FP32, same shape as input
at::Tensor engram_fused_weight_meta(const at::Tensor& wh_data, const at::Tensor& we_data)
{
    return at::empty_like(wh_data, wh_data.options().dtype(at::kFloat));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("engram_fused_weight", &engram_fused_weight_meta);
}

} // namespace
