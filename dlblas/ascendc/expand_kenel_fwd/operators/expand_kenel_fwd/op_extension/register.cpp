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

// 算子签名: expand_kenel_fwd(Tensor x, int mhc_mult) -> Tensor
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("expand_kenel_fwd(Tensor x, int mhc_mult) -> Tensor");
}

// NPU 后端实现绑定
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("expand_kenel_fwd", TORCH_FN(ascend_kernel::expand_kenel_fwd_torch));
}

// Meta 后端（torch.compile / fx 需要）
at::Tensor expand_kenel_fwd_meta(const at::Tensor& x, int64_t mhc_mult)
{
    // 输入形状: (..., H), 输出形状: (..., mhc_mult, H)
    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end() - 1);
    outShape.push_back(mhc_mult);
    outShape.push_back(x.size(-1));
    return at::empty(outShape, x.options());
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("expand_kenel_fwd", &expand_kenel_fwd_meta);
}

} // namespace
