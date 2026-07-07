/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor expand_kenel_bwd_torch(const at::Tensor& o_grad);

} // namespace ascend_kernel

#endif // OPS_H
