/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * register.cpp - TORCH_LIBRARY registration for sparse_attn
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sparse_attn(Tensor q, Tensor kv, Tensor attn_sink, "
          "Tensor topk_idxs, Scalar softmax_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sparse_attn", TORCH_FN(ascend_kernel::sparse_attn_torch));
}

at::Tensor sparse_attn_meta(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& attn_sink,
    const at::Tensor& topk_idxs,
    const c10::Scalar& softmax_scale)
{
    return at::empty_like(q);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sparse_attn", &sparse_attn_meta);
}

} // namespace
