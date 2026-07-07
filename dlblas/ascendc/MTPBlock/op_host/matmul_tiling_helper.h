/**
 * MatmulApiTiling C++ Helper
 *
 * Provides C-linkage functions for computing TCubeTiling on the host side.
 * This file is compiled as CXX (not ASC) and linked with both the
 * direct-invoke host and the torch extension.
 *
 * NpuArch: DAV_2201, CANN 9.0.0
 */

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute TCubeTiling for a simple MatMul C = A [M, K] × B^T [N, K]
 * A: half (bf16), B: half (bf16), C: float
 *
 * @param tilingBuf  Output buffer to receive TCubeTiling data (caller allocates)
 * @param bufSize    Size of tilingBuf in bytes
 * @param M          M dimension of A
 * @param N          N dimension of B^T (output columns)
 * @param K          K dimension (inner dimension)
 * @param isTransA   Whether A is transposed
 * @param isTransB   Whether B is transposed
 * @return           Size of TCubeTiling written, or 0 on error
 */
uint32_t ComputeMatmulTiling(
    void* tilingBuf,
    uint32_t bufSize,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool isTransA,
    bool isTransB);

#ifdef __cplusplus
}
#endif
