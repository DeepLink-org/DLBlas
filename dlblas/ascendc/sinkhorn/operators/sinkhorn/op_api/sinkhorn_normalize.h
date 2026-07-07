/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize.h
 * @brief ACLNN L0 API for Sinkhorn Normalize operator
 *
 * L0 API: shape inference and kernel dispatch.
 */

#ifndef OP_API_INC_LEVEL0_SINKHORN_NORMALIZE_H_
#define OP_API_INC_LEVEL0_SINKHORN_NORMALIZE_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* SinkhornNormalize(const aclTensor* x, float eps, int64_t repeat,
                                    aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SINKHORN_NORMALIZE_H_
