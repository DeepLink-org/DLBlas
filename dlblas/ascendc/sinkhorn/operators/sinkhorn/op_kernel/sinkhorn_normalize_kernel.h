/**
 * @file sinkhorn_normalize_kernel.h
 * @brief Sinkhorn Normalize Ascend C Kernel - Tiling Data & Function Declaration
 *
 * Target: Ascend 910B2 (DAV_2201), CANN 9.0.0
 */

#ifndef SINKHORN_NORMALIZE_KERNEL_H
#define SINKHORN_NORMALIZE_KERNEL_H

#include <cstdint>

struct SinkhornNormalizeTilingData {
    uint32_t total_matrices;       // total number of 4x4 matrices (1024)
    uint32_t matrices_per_core;    // matrices assigned to this core
    uint32_t matrix_start_offset;  // starting matrix index for this core
    uint32_t repeat;               // number of repeat iterations (10)
    float    eps;                  // epsilon value (1e-6)
};

#endif // SINKHORN_NORMALIZE_KERNEL_H
