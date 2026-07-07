/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file aclnn_sinkhorn_normalize.cpp
 * @brief ACLNN L2 API implementation for Sinkhorn Normalize operator
 */

#include "aclnn_sinkhorn_normalize.h"
#include "sinkhorn_normalize.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"

using namespace op;

extern "C" aclnnStatus aclnnSinkhornNormalizeGetWorkspaceSize(
    const aclTensor* x,
    float eps,
    int64_t repeat,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSinkhornNormalize, DFX_IN(x), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor* opResult = l0op::SinkhornNormalize(x, eps, repeat, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSinkhornNormalize(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSinkhornNormalize);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
