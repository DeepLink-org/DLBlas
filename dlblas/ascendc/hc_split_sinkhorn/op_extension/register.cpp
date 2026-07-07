/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("hc_split_sinkhorn(Tensor mixes, int hc_mult, int sinkhorn_iters, "
          "float eps, Tensor hc_scale, Tensor hc_base, "
          "Tensor(a!) pre, Tensor(b!) post, Tensor(c!) comb) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hc_split_sinkhorn", TORCH_FN(ascend_kernel::hc_split_sinkhorn_torch));
}

void hc_split_sinkhorn_meta(
    const at::Tensor& mixes,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& pre,
    at::Tensor& post,
    at::Tensor& comb)
{
    // Meta 实现无需实际计算，仅用于 torch.compile / fx
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("hc_split_sinkhorn", &hc_split_sinkhorn_meta);
}

} // namespace
