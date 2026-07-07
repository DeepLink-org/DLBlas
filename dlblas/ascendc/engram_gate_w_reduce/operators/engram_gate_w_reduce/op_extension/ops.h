#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> engram_gate_w_reduce_torch(
    const at::Tensor& grad_w_partial,
    const at::Tensor& weight_hidden,
    const at::Tensor& weight_embed,
    const at::Tensor& grad_weight_hidden,
    const at::Tensor& grad_weight_embed);

} // namespace ascend_kernel

#endif // OPS_H
