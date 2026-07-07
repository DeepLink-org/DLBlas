#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("engram_gate_fwd(Tensor hidden_states, Tensor k, Tensor v, "
          "Tensor weight_hidden, Tensor weight_embed, "
          "float clamp_value, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("engram_gate_fwd", TORCH_FN(ascend_kernel::engram_gate_fwd_torch));
}

// Meta implementation for torch.compile / FX
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> engram_gate_fwd_meta(
    const at::Tensor& hidden_states,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& weight_hidden,
    const at::Tensor& weight_embed,
    double clamp_value,
    double eps)
{
    int64_t num_tokens  = hidden_states.size(0);
    int64_t hc_mult     = hidden_states.size(1);
    int64_t hidden_size = hidden_states.size(2);

    return std::make_tuple(
        at::empty_like(hidden_states),
        at::empty({num_tokens, hc_mult}, hidden_states.options().dtype(at::kFloat)),
        at::empty({num_tokens, hc_mult}, hidden_states.options().dtype(at::kFloat)),
        at::empty({num_tokens, hc_mult}, hidden_states.options().dtype(at::kFloat)),
        at::empty({num_tokens, hc_mult}, hidden_states.options().dtype(at::kFloat))
    );
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("engram_gate_fwd", &engram_gate_fwd_meta);
}

} // namespace
