/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize_tiling.cpp
 * @brief SinkhornNormalize tiling implementation (arch22 / DAV_2201)
 *
 * Multi-core partitioning:
 *   Total 4x4 matrices = product of all dims except last two (which are always 4).
 *   Evenly distribute matrices across AI Cores.
 *   Each core's tiling data contains: start offset, count, total, repeat, eps.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/sinkhorn_normalize_tiling_data.h"
#include "../../op_kernel/arch22/sinkhorn_normalize_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

// Get platform info: UB size and AI Core count
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Get shape and attribute info
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context,
                                          int64_t* totalMatrices, float* eps, int64_t* repeat)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    // dtype check
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT, OP_LOGE(context, "Only float32 supported"), return ge::GRAPH_FAILED);

    // Compute total number of 4x4 matrices: product of all dims except last two.
    // For [B, S, 4, 4]: totalMatrices = B * S.
    int64_t dimNum = inputShapeX.GetDimNum();
    if (dimNum < 2) {
        OP_LOGE(context, "Input must have at least 2 dims, got %ld", dimNum);
        return ge::GRAPH_FAILED;
    }
    // Verify last two dims are 4
    int64_t lastDim  = inputShapeX.GetDim(dimNum - 1);
    int64_t lastDim2 = inputShapeX.GetDim(dimNum - 2);
    OP_CHECK_IF(lastDim != 4 || lastDim2 != 4,
        OP_LOGE(context, "Last two dims must be 4, got [%ld, %ld]", lastDim2, lastDim),
        return ge::GRAPH_FAILED);

    *totalMatrices = 1;
    for (int64_t i = 0; i < dimNum - 2; i++) {
        *totalMatrices *= inputShapeX.GetDim(i);
    }

    // Get attributes via RuntimeAttrs
    *eps = 1e-6f;
    *repeat = 10;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsPtr = attrs->GetFloat(0);
        if (epsPtr != nullptr) {
            *eps = *epsPtr;
        }
        const int64_t* repeatPtr = attrs->GetInt(1);
        if (repeatPtr != nullptr) {
            *repeat = *repeatPtr;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Tiling entry point
static ge::graphStatus SinkhornNormalizeTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Get shape and attribute info
    int64_t totalMatrices;
    float eps;
    int64_t repeat;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalMatrices, &eps, &repeat) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. Set tiling data
    SinkhornNormalizeTilingData* tiling = context->GetTilingData<SinkhornNormalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SinkhornNormalizeTilingData), 0, sizeof(SinkhornNormalizeTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Empty tensor check
    if (totalMatrices == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT));
        return ge::GRAPH_SUCCESS;
    }

    // Multi-core partitioning: distribute matrices across AI Cores
    // Limit cores to total matrices (don't use more cores than matrices)
    int64_t usedCoreNum = (totalMatrices < coreNum) ? totalMatrices : coreNum;
    int64_t baseMatricesPerCore = totalMatrices / usedCoreNum;
    int64_t remainder = totalMatrices % usedCoreNum;

    // Set shared tiling data: each block computes its own partition via GetBlockIdx()
    tiling->total_matrices       = totalMatrices;
    tiling->matrices_per_core_base = baseMatricesPerCore;
    tiling->remainder            = remainder;
    tiling->repeat               = repeat;
    tiling->eps                  = eps;

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey (only float32 supported)
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSinkhornNormalize([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SinkhornNormalizeCompileInfo {};

// Tiling registration
IMPL_OP_OPTILING(SinkhornNormalize)
    .Tiling(SinkhornNormalizeTilingFunc)
    .TilingParse<SinkhornNormalizeCompileInfo>(TilingParseForSinkhornNormalize);

} // namespace optiling
