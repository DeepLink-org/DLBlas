#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("engram_gate_w_reduce(Tensor grad_w_partial, Tensor weight_hidden, "
          "Tensor weight_embed, Tensor grad_weight_hidden, Tensor grad_weight_embed) "
          "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("engram_gate_w_reduce", TORCH_FN(ascend_kernel::engram_gate_w_reduce_torch));
}

// Meta backend 注册（torch.compile / fx 需要）
std::tuple<at::Tensor, at::Tensor> engram_gate_w_reduce_meta(
    const at::Tensor& grad_w_partial,
    const at::Tensor& weight_hidden,
    const at::Tensor& weight_embed,
    const at::Tensor& grad_weight_hidden,
    const at::Tensor& grad_weight_embed)
{
    return {at::empty_like(grad_weight_hidden), at::empty_like(grad_weight_embed)};
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("engram_gate_w_reduce", &engram_gate_w_reduce_meta);
}

} // namespace
