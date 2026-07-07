/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

void hc_split_sinkhorn_torch(
    const at::Tensor& mixes,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& pre,
    at::Tensor& post,
    at::Tensor& comb);

} // namespace ascend_kernel

#endif // OPS_H
