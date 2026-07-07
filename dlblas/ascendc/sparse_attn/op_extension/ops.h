#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor sparse_attn_torch(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& attn_sink,
    const at::Tensor& topk_idxs,
    double softmax_scale);

} // namespace ascend_kernel

#endif // OPS_H
