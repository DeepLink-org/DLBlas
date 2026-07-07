#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor norm_fn_torch(
    const at::Tensor& residual,
    const at::Tensor& mhc_fn,
    const c10::optional<at::Tensor>& mhc_norm_weight,
    double mhc_norm_eps);

} // namespace ascend_kernel

#endif // OPS_H
