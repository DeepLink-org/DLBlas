/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize_infershape.cpp
 * @brief SinkhornNormalize operator shape inference
 *
 * Output shape = Input shape (element-wise in 4D space).
 * The last two dimensions must be exactly 4.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4SinkhornNormalize(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // Output shape = input shape (same shape, float32)
    *outputShape = *inputShape;

    // Validate: last two dims must be 4
    int64_t dimNum = inputShape->GetDimNum();
    if (dimNum >= 2) {
        int64_t lastDim  = inputShape->GetDim(dimNum - 1);
        int64_t lastDim2 = inputShape->GetDim(dimNum - 2);
        OP_CHECK_IF(lastDim != 4 || lastDim2 != 4,
            OP_LOGE(context, "SinkhornNormalize: last two dims must be 4, got [%ld, %ld]",
                    lastDim2, lastDim),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "SinkhornNormalize: input must have at least 2 dims, got %ld", dimNum);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SinkhornNormalize).InferShape(InferShape4SinkhornNormalize);

} // namespace ops
