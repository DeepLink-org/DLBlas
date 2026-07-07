#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor head_compute_mix_fwd_torch(
    const at::Tensor& input_mix,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    double mhc_pre_eps);

} // namespace ascend_kernel

#endif // OPS_H
