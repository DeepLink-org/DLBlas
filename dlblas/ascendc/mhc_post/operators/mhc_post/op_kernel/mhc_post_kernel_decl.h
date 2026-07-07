/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

// Kernel function declaration - shared between host and kernel
#pragma once

#include "kernel_operator.h"
#include "mhc_post_tiling.h"

extern "C" __global__ __vector__ void mhc_post_kernel(
    GM_ADDR x, GM_ADDR residual, GM_ADDR post_layer_mix,
    GM_ADDR comb_res_mix, GM_ADDR output, GM_ADDR tiling);
