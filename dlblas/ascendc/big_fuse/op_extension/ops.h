/**
 * big_fuse PyTorch Extension - Function declarations
 */

#pragma once

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor big_fuse_torch(
    const at::Tensor& residual,
    const at::Tensor& fn,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base);

} // namespace ascend_kernel
