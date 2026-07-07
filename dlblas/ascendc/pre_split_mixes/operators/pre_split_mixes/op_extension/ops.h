/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor> pre_split_mixes_torch(
    const at::Tensor& input_mixes,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base,
    int64_t mhc_mult,
    double mhc_pre_eps,
    double mhc_post_mult_value);

} // namespace ascend_kernel

#endif // OPS_H
