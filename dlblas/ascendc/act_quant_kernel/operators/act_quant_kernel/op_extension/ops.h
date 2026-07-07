/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

// act_quant_kernel: 1 input -> 2 outputs
// x (bf16/fp16) → x_q (fp8_e4m3fn), x_s (fp32)
std::tuple<at::Tensor, at::Tensor> act_quant_kernel_torch(
    const at::Tensor& x, int64_t group_size, double eps, bool scale_ue8m0);

} // namespace ascend_kernel

#endif // OPS_H
