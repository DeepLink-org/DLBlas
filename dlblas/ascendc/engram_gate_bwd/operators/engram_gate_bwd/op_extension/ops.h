#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::vector<at::Tensor> engram_gate_bwd_torch(
    const at::Tensor& grad_out, const at::Tensor& x,
    const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& wh, const at::Tensor& we);

} // namespace ascend_kernel

#endif // OPS_H
