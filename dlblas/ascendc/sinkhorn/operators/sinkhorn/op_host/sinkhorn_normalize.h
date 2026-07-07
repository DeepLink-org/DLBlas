/**
 * @file sinkhorn_normalize.h
 * @brief Sinkhorn Normalize Host Operator Declaration
 */

#ifndef SINKHORN_NORMALIZE_HOST_H
#define SINKHORN_NORMALIZE_HOST_H

#include <cstdint>
#include <vector>

/**
 * @brief Launch the Sinkhorn Normalize operator on Ascend NPU.
 *
 * @param input            Pointer to input tensor data on host (shape: [B, S, 4, 4], float32)
 * @param output           Pointer to output tensor data on host (same shape, float32)
 * @param B                Batch dimension (currently must be 1)
 * @param S                Sequence dimension (number of 4x4 matrices)
 * @param repeat           Number of Sinkhorn iterations (default 10)
 * @param eps              Epsilon value (default 1e-6)
 * @param kernel_bin_path  Path to the compiled kernel binary (.o file)
 * @return                 0 on success, non-zero on failure
 */
int sinkhorn_normalize_launch(const float* input, float* output,
                              uint32_t B, uint32_t S,
                              uint32_t repeat = 10, float eps = 1e-6f,
                              const char* kernel_bin_path = nullptr);

#endif // SINKHORN_NORMALIZE_HOST_H
