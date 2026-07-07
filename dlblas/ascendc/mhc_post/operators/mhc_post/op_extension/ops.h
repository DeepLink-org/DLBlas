/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#pragma once

#include <torch/torch.h>

namespace ascend_kernel {

at::Tensor mhc_post_torch(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix);

}  // namespace ascend_kernel
