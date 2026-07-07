/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor> head_compute_mix_bwd_torch(
    const at::Tensor& input_mix,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    const at::Tensor& grad_out);

} // namespace ascend_kernel

#endif // OPS_H
