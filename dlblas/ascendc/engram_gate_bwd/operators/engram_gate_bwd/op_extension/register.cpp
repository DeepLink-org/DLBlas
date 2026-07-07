/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// Operator registration stub
// ============================================================================

#include <torch/extension.h>

namespace ascend_kernel {
    // Forward declaration
    at::Tensor engram_gate_bwd_torch(
        at::Tensor grad_out, at::Tensor x, at::Tensor k, at::Tensor v,
        at::Tensor wh, at::Tensor we, double clamp_value, double eps);
}

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("engram_gate_bwd(Tensor go, Tensor x, Tensor k, Tensor v, "
          "Tensor wh, Tensor we, float clamp_value, float eps) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("engram_gate_bwd", TORCH_FN(ascend_kernel::engram_gate_bwd_torch));
}

// Meta backend
std::vector<at::Tensor> engram_gate_bwd_meta(
    const at::Tensor& go, const at::Tensor& x, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& wh, const at::Tensor& we,
    double clamp_value, double eps)
{
    return {at::empty_like(x), at::empty_like(k), at::empty_like(v),
            at::empty_like(wh), at::empty_like(we)};
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("engram_gate_bwd", &engram_gate_bwd_meta);
}
