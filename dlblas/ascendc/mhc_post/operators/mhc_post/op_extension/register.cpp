/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

// TORCH_LIBRARY registration for mhc_post operator

#include "acl/acl.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "ops.h"

// NPU backend meta function (shape inference for torch.compile / fx)
static at::Tensor mhc_post_meta(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix)
{
    // output shape: (n0, n1, M=4, h) bf16
    auto n0 = x.size(0);
    auto n1 = x.size(1);
    auto h  = x.size(2);
    auto M  = residual.size(2);
    return at::empty({n0, n1, M, h}, x.options());
}

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("mhc_post(Tensor x, Tensor residual, Tensor post_layer_mix, "
          "Tensor comb_res_mix) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("mhc_post", TORCH_FN(ascend_kernel::mhc_post_torch));
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("mhc_post", &mhc_post_meta);
}
