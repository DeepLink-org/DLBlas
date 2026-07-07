#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor sinkhorn_torch(const at::Tensor& x);

} // namespace ascend_kernel

#endif // OPS_H
