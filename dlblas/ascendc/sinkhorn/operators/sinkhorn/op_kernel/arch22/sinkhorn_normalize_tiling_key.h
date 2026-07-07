/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize_tiling_key.h
 * @brief Tiling template parameters for Sinkhorn Normalize operator
 *
 * Only float32 is supported. The operator is memory-bound and tiny (4x4 matrices),
 * so double-buffering is not beneficial.
 */

#ifndef SINKHORN_NORMALIZE_TILING_KEY_H
#define SINKHORN_NORMALIZE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SinkhornNormalize,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    ),
);

#endif // SINKHORN_NORMALIZE_TILING_KEY_H
