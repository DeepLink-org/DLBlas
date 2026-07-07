#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
engram_gate_fwd_torch(
    const at::Tensor& hidden_states,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& weight_hidden,
    const at::Tensor& weight_embed,
    double clamp_value,
    double eps);

} // namespace ascend_kernel

#endif // OPS_H
