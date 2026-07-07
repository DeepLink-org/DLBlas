/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize_tiling_data.h
 * @brief Tiling data structure for Sinkhorn Normalize operator
 */

#ifndef SINKHORN_NORMALIZE_TILING_DATA_H
#define SINKHORN_NORMALIZE_TILING_DATA_H

#include <cstdint>

struct SinkhornNormalizeTilingData {
    int64_t total_matrices = 0;          // B * S, total number of 4x4 matrices
    int64_t matrices_per_core_base = 0;  // base matrices per core (floor division)
    int64_t remainder = 0;               // first N cores get +1 matrix
    int64_t repeat = 10;                 // number of Sinkhorn iterations
    float   eps = 1e-6f;                 // epsilon for numerical stability
};

#endif // SINKHORN_NORMALIZE_TILING_DATA_H
