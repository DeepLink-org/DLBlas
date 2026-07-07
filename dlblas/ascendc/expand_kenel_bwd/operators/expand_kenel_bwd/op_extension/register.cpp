/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

// 算子签名: 单输入 o_grad (n0, n1, mhc_mult, h) → 输出 (n0, n1, h)
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("expand_kenel_bwd(Tensor o_grad) -> Tensor");
}

// NPU 后端实现绑定
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("expand_kenel_bwd", TORCH_FN(ascend_kernel::expand_kenel_bwd_torch));
}

// Meta 后端注册 (torch.compile / fx 需要)
at::Tensor expand_kenel_bwd_meta(const at::Tensor& o_grad)
{
    // 输出 shape: (n0, n1, h) — 去掉 dim=-2
    auto sizes = o_grad.sizes();
    int64_t n0 = sizes[0];
    int64_t n1 = sizes[1];
    int64_t h  = sizes[3];
    return at::empty({n0, n1, h}, o_grad.options());
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("expand_kenel_bwd", &expand_kenel_bwd_meta);
}

} // namespace
