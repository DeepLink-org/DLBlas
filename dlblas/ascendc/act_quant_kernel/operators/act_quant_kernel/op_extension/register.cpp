/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("act_quant_kernel(Tensor x, int group_size, float eps, bool scale_ue8m0) -> (Tensor x_q, Tensor x_s)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("act_quant_kernel", TORCH_FN(ascend_kernel::act_quant_kernel_torch));
}

// Meta backend for torch.compile / fx
std::tuple<at::Tensor, at::Tensor> act_quant_kernel_meta(
    const at::Tensor& x, int64_t group_size, double eps, bool scale_ue8m0)
{
    int64_t N = x.size(-1);
    auto sShape = x.sizes().vec();
    sShape.back() = N / group_size;

    auto x_q = at::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));
    auto x_s = at::empty(sShape, x.options().dtype(at::kFloat));

    return std::make_tuple(x_q, x_s);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("act_quant_kernel", &act_quant_kernel_meta);
}

} // namespace
