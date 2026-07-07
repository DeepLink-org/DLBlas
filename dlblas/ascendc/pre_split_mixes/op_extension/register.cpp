/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// 算子签名 (返回 3 个 Tensor)
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("pre_split_mixes(Tensor input_mixes, Tensor mhc_scale, Tensor mhc_base, "
          "int mhc_mult, float mhc_pre_eps, float mhc_post_mult_value) "
          "-> (Tensor pre_mix, Tensor post_mix, Tensor comb_mix)");
}

// NPU 后端绑定
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("pre_split_mixes", TORCH_FN(ascend_kernel::pre_split_mixes_torch));
}

// Meta 实现 (torch.compile / fx 需要)
std::tuple<at::Tensor, at::Tensor, at::Tensor> pre_split_mixes_meta(
    const at::Tensor& input_mixes,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    int64_t mhc_mult,
    double mhc_pre_eps,
    double mhc_post_mult_value)
{
    auto batch = input_mixes.size(0);
    auto seq_len = input_mixes.size(1);
    auto m = static_cast<int64_t>(mhc_mult);
    return std::make_tuple(
        at::empty({batch, seq_len, m}, input_mixes.options()),
        at::empty({batch, seq_len, m}, input_mixes.options()),
        at::empty({batch, seq_len, m * m}, input_mixes.options())
    );
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("pre_split_mixes", &pre_split_mixes_meta);
}

} // namespace
