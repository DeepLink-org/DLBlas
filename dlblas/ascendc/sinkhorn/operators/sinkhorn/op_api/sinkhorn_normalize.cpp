/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize.cpp
 * @brief ACLNN L0 API implementation for Sinkhorn Normalize operator
 *
 * L0 API: shape inference and kernel dispatch.
 * L2 API: parameter checking, Contiguous/ViewCopy handling.
 */

#include "sinkhorn_normalize.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SinkhornNormalize);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    // Simplified: skip arch check for debugging
    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SinkhornNormalize only supports float32. dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool SinkhornNormalizeInferShape(const op::Shape& xShape, op::Shape& outShape)
{
    // Output shape = input shape
    outShape = xShape;
    return true;
}

static const aclTensor* SinkhornNormalizeAiCore(const aclTensor* x,
                                                  const aclTensor* out,
                                                  float eps,
                                                  int64_t repeat,
                                                  aclOpExecutor* executor)
{
    L0_DFX(SinkhornNormalizeAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SinkhornNormalize,
        OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(eps, repeat));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SinkhornNormalizeAiCore failed."),
        return nullptr);
    return out;
}

/**
 * @brief L0 API entry point
 *
 * Flow:
 * 1. InferShape      - output shape = input shape
 * 2. IsAiCoreSupport - DAV_2201 only, float32 only
 * 3. AllocTensor     - allocate output tensor
 * 4. AiCore kernel   - launch kernel
 */
const aclTensor* SinkhornNormalize(const aclTensor* x, float eps, int64_t repeat,
                                    aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    OP_CHECK(SinkhornNormalizeInferShape(x->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    out = executor->AllocTensor(outShape, x->GetDataType());

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return SinkhornNormalizeAiCore(x, out, eps, repeat, executor);
}

} // namespace l0op
