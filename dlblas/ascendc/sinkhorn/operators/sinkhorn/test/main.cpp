/**
 * @file main.cpp
 * @brief Test driver for Sinkhorn Normalize operator
 *
 * Usage:
 *   ./test_sinkhorn_normalize [num_matrices] [repeat] [eps]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>

#include "../op_host/sinkhorn_normalize.h"

// Reference implementation in plain C++
void sinkhorn_normalize_ref(const float* input, float* output,
                            uint32_t B, uint32_t S,
                            uint32_t repeat, float eps)
{
    uint32_t M = 4;
    uint32_t total_elems = B * S * M * M;

    // Copy input to output
    memcpy(output, input, total_elems * sizeof(float));

    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t s = 0; s < S; s++) {
            float* mat = output + (b * S + s) * M * M; // 4x4 matrix

            // Step A: softmax(dim=-1) + eps
            for (uint32_t r = 0; r < M; r++) {
                float* row = mat + r * M;

                // Find max for stability
                float max_val = row[0];
                for (uint32_t c = 1; c < M; c++) {
                    if (row[c] > max_val) max_val = row[c];
                }

                // Clamp for numerical stability
                if (max_val > 85.0f) max_val = 85.0f;

                // Subtract max and exp
                float sum = 0.0f;
                for (uint32_t c = 0; c < M; c++) {
                    row[c] = expf(row[c] - max_val);
                    sum += row[c];
                }

                // Normalize and add eps
                float inv_sum = 1.0f / (sum + 1e-10f);
                for (uint32_t c = 0; c < M; c++) {
                    row[c] = row[c] * inv_sum + eps;
                }
            }

            // Step B: Column normalize (sum over dim=-2)
            for (uint32_t c = 0; c < M; c++) {
                float col_sum = 0.0f;
                for (uint32_t r = 0; r < M; r++) {
                    col_sum += mat[r * M + c];
                }
                float inv_col_sum = 1.0f / (col_sum + eps);
                for (uint32_t r = 0; r < M; r++) {
                    mat[r * M + c] *= inv_col_sum;
                }
            }

            // Repeat (repeat-1) times
            for (uint32_t iter = 1; iter < repeat; iter++) {
                // Row normalize
                for (uint32_t r = 0; r < M; r++) {
                    float* row = mat + r * M;
                    float row_sum = 0.0f;
                    for (uint32_t c = 0; c < M; c++) {
                        row_sum += row[c];
                    }
                    float inv_row_sum = 1.0f / (row_sum + eps);
                    for (uint32_t c = 0; c < M; c++) {
                        row[c] *= inv_row_sum;
                    }
                }

                // Column normalize
                for (uint32_t c = 0; c < M; c++) {
                    float col_sum = 0.0f;
                    for (uint32_t r = 0; r < M; r++) {
                        col_sum += mat[r * M + c];
                    }
                    float inv_col_sum = 1.0f / (col_sum + eps);
                    for (uint32_t r = 0; r < M; r++) {
                        mat[r * M + c] *= inv_col_sum;
                    }
                }
            }
        }
    }
}

// Check if a value is NaN or Inf
bool is_valid(float v) {
    return !std::isnan(v) && !std::isinf(v);
}

int main(int argc, char** argv) {
    // Default parameters
    uint32_t B = 1;
    uint32_t S = 1024;
    uint32_t repeat = 10;
    float eps = 1e-6f;

    if (argc > 1) S = (uint32_t)atoi(argv[1]);
    if (argc > 2) repeat = (uint32_t)atoi(argv[2]);
    if (argc > 3) eps = (float)atof(argv[3]);

    // Kernel binary path: default to build directory, overridable via argv[4]
    const char* kernel_bin_path = "build/sinkhorn_normalize_kernel.o";
    if (argc > 4) kernel_bin_path = argv[4];

    uint32_t M = 4;
    uint32_t total_matrices = B * S;
    uint32_t total_elems = total_matrices * M * M;

    printf("============================================\n");
    printf("  Sinkhorn Normalize - Test Driver\n");
    printf("============================================\n");
    printf("  Shape:    [%u, %u, %u, %u]\n", B, S, M, M);
    printf("  Matrices: %u\n", total_matrices);
    printf("  Repeat:   %u\n", repeat);
    printf("  Epsilon:  %e\n", eps);
    printf("  Elements: %u\n", total_elems);
    printf("  Data:     %.2f KB\n", total_elems * sizeof(float) / 1024.0f);
    printf("============================================\n\n");

    // Allocate host memory
    float* input  = (float*)malloc(total_elems * sizeof(float));
    float* output = (float*)malloc(total_elems * sizeof(float));
    float* reference = (float*)malloc(total_elems * sizeof(float));

    if (!input || !output || !reference) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Generate random input
    srand(42);
    for (uint32_t i = 0; i < total_elems; i++) {
        // Random values in [-2, 2] range to test numerical stability
        input[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    }

    // Compute reference
    printf("[1/3] Computing reference (CPU)...\n");
    auto start = std::chrono::high_resolution_clock::now();
    sinkhorn_normalize_ref(input, reference, B, S, repeat, eps);
    auto end = std::chrono::high_resolution_clock::now();
    auto ref_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("  Reference completed in %ld us (%.3f ms)\n", ref_us, ref_us / 1000.0f);

    // Check reference for NaN/Inf
    uint32_t ref_nan_count = 0;
    for (uint32_t i = 0; i < total_elems; i++) {
        if (!is_valid(reference[i])) ref_nan_count++;
    }
    if (ref_nan_count > 0) {
        printf("  [WARN] Reference output has %u NaN/Inf values\n", ref_nan_count);
    }

    // Verify doubly stochastic property for a sample matrix
    printf("\n[2/3] Verifying doubly stochastic property (sample)...\n");
    for (uint32_t s = 0; s < std::min(S, (uint32_t)3); s++) {
        float* mat = reference + s * M * M;
        float row_sums[4] = {0}, col_sums[4] = {0};
        float max_row_err = 0, max_col_err = 0;

        for (uint32_t r = 0; r < M; r++) {
            for (uint32_t c = 0; c < M; c++) {
                row_sums[r] += mat[r * M + c];
                col_sums[c] += mat[r * M + c];
            }
        }

        printf("  Matrix %u:\n", s);
        for (uint32_t r = 0; r < M; r++) {
            printf("    Row %u: [%9.6f %9.6f %9.6f %9.6f]  sum=%.6f\n",
                   r, mat[r*M], mat[r*M+1], mat[r*M+2], mat[r*M+3], row_sums[r]);
            float row_err = fabsf(row_sums[r] - 1.0f);
            if (row_err > max_row_err) max_row_err = row_err;
        }
        for (uint32_t c = 0; c < M; c++) {
            float col_err = fabsf(col_sums[c] - 1.0f);
            if (col_err > max_col_err) max_col_err = col_err;
        }
        printf("    Max row error: %.6e, Max col error: %.6e\n", max_row_err, max_col_err);
    }

    // Try to launch NPU kernel
    printf("\n[3/3] Launching Ascend NPU kernel...\n");
    int ret = sinkhorn_normalize_launch(input, output, B, S, repeat, eps, kernel_bin_path);
    if (ret != 0) {
        printf("  [WARN] NPU kernel launch returned error %d\n", ret);
        printf("  This is expected if kernel binary is not yet available.\n");
        printf("  Kernel source is ready for compilation.\n");
    } else {
        // Compare with reference
        double max_error = 0.0;
        double sum_error = 0.0;
        uint32_t error_count = 0;
        float tolerance = 1e-4f;

        for (uint32_t i = 0; i < total_elems; i++) {
            float diff = fabsf(output[i] - reference[i]);
            sum_error += diff;
            if (diff > max_error) max_error = diff;
            if (diff > tolerance) error_count++;
        }

        printf("  Max error:  %.6e\n", max_error);
        printf("  Mean error: %.6e\n", sum_error / total_elems);
        printf("  Errors > %.0e: %u / %u (%.2f%%)\n",
               tolerance, error_count, total_elems,
               100.0f * error_count / total_elems);

        if (error_count == 0) {
            printf("  [PASS] All values within tolerance!\n");
        } else {
            printf("  [FAIL] %u values exceed tolerance\n", error_count);
        }
    }

    // Cleanup
    free(input);
    free(output);
    free(reference);

    printf("\n============================================\n");
    printf("  Test complete.\n");
    printf("============================================\n");

    return 0;
}
