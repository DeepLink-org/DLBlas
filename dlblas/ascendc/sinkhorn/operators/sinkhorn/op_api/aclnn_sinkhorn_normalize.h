/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file aclnn_sinkhorn_normalize.h
 * @brief ACLNN L2 API header for Sinkhorn Normalize operator
 *
 * Two-stage design:
 * - aclnnSinkhornNormalizeGetWorkspaceSize: compute workspace size and create executor
 * - aclnnSinkhornNormalize: execute computation
 */

#ifndef ACLNN_SINKHORN_NORMALIZE_H_
#define ACLNN_SINKHORN_NORMALIZE_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size needed for SinkhornNormalize
 * @param x [in] Input tensor (float32, last two dims must be 4)
 * @param eps [in] Epsilon value for numerical stability
 * @param repeat [in] Number of Sinkhorn iterations
 * @param out [in] Output tensor
 * @param workspaceSize [out] Required workspace size
 * @param executor [out] Op executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnSinkhornNormalizeGetWorkspaceSize(
    const aclTensor *x,
    float eps,
    int64_t repeat,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute SinkhornNormalize computation
 * @param workspace [in] Workspace memory
 * @param workspaceSize [in] Workspace size
 * @param executor [in] Op executor
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnSinkhornNormalize(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SINKHORN_NORMALIZE_H_
