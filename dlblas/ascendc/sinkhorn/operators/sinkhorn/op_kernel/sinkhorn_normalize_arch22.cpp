/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize_arch22.cpp
 * @brief Sinkhorn Normalize kernel entry (arch22 / DAV_2201 / Ascend 910B2)
 */

#include "arch22/sinkhorn_normalize.h"

template <typename D_T_X>
__global__ __aicore__ void sinkhorn_normalize(GM_ADDR input, GM_ADDR output,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SinkhornNormalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(SinkhornNormalizeTilingData, tilingData, tiling);
    NsSinkhornNormalize::SinkhornNormalize<D_T_X> op;
    op.Init(input, output, &tilingData);
    op.Process();
}
