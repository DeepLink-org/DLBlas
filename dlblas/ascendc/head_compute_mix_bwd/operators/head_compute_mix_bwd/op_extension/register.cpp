/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("head_compute_mix_bwd(Tensor input_mix, Tensor mhc_scale, "
          "Tensor mhc_base, Tensor grad_out) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("head_compute_mix_bwd", TORCH_FN(ascend_kernel::head_compute_mix_bwd_torch));
}

// Meta backend for torch.compile / fx
std::tuple<at::Tensor, at::Tensor, at::Tensor> head_compute_mix_bwd_meta(
    const at::Tensor& input_mix,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    const at::Tensor& grad_out)
{
    auto grad_input_mix = at::empty_like(input_mix);
    auto grad_mhc_scale = at::empty({1}, input_mix.options());
    auto grad_mhc_base = at::empty({4}, input_mix.options());
    return std::make_tuple(grad_input_mix, grad_mhc_scale, grad_mhc_base);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("head_compute_mix_bwd", &head_compute_mix_bwd_meta);
}

} // namespace
