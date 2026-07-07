#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor apply_mix_torch(const at::Tensor& x, const at::Tensor& mix);

} // namespace ascend_kernel

#endif // OPS_H
