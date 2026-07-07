/**
 * Precision test for Sinkhorn Normalize custom operator.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnn_sinkhorn_normalize.h"

static void sinkhorn_golden(const float* input, float* output,
                            int B, int S, int repeat, float eps) {
    int M = 4;
    int total = B * S * M * M;
    memcpy(output, input, total * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            float* mat = output + (b * S + s) * M * M;
            for (int r = 0; r < M; r++) {
                float* row = mat + r * M;
                float max_val = row[0];
                for (int c = 1; c < M; c++) if (row[c] > max_val) max_val = row[c];
                if (max_val > 85.0f) max_val = 85.0f;
                float sum = 0.0f;
                for (int c = 0; c < M; c++) { row[c] = expf(row[c] - max_val); sum += row[c]; }
                float inv_sum = 1.0f / (sum + eps);
                for (int c = 0; c < M; c++) row[c] = row[c] * inv_sum + eps;
            }
            for (int c = 0; c < M; c++) {
                float col_sum = 0.0f;
                for (int r = 0; r < M; r++) col_sum += mat[r * M + c];
                float inv = 1.0f / (col_sum + eps);
                for (int r = 0; r < M; r++) mat[r * M + c] *= inv;
            }
            for (int iter = 1; iter < repeat; iter++) {
                for (int r = 0; r < M; r++) {
                    float* row = mat + r * M; float row_sum = 0.0f;
                    for (int c = 0; c < M; c++) row_sum += row[c];
                    float inv = 1.0f / (row_sum + eps);
                    for (int c = 0; c < M; c++) row[c] *= inv;
                }
                for (int c = 0; c < M; c++) {
                    float col_sum = 0.0f;
                    for (int r = 0; r < M; r++) col_sum += mat[r * M + c];
                    float inv = 1.0f / (col_sum + eps);
                    for (int r = 0; r < M; r++) mat[r * M + c] *= inv;
                }
            }
        }
    }
}

int run_test(int S, int repeat, float eps, int seed) {
    int B = 1, M = 4;
    int total_matrices = B * S;
    int total = total_matrices * M * M;

    printf("=== Sinkhorn Precision Test: [%d,%d,4,4] repeat=%d eps=%.1e ===\n",
           B, S, repeat, eps);

    // Generate random input
    srand(seed);
    std::vector<float> input(total);
    for (int i = 0; i < total; i++)
        input[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;

    // CPU golden
    printf("Computing CPU reference...");
    fflush(stdout);
    std::vector<float> golden(total);
    sinkhorn_golden(input.data(), golden.data(), B, S, repeat, eps);
    printf(" done\n");

    // ACL init
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // Device memory
    void *d_in = nullptr, *d_out = nullptr;
    size_t data_size = total * sizeof(float);
    aclrtMalloc(&d_in, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_out, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_in, data_size, input.data(), data_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // Tensors
    std::vector<int64_t> shape = {1, (int64_t)S, 4, 4};
    std::vector<int64_t> strides = {(int64_t)(S*16), 16, 4, 1};
    aclTensor *t_in = aclCreateTensor(shape.data(), 4, ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, shape.data(), 4, d_in);
    aclTensor *t_out = aclCreateTensor(shape.data(), 4, ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, shape.data(), 4, d_out);

    // Execute
    printf("Executing NPU kernel...");
    fflush(stdout);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    int ret = aclnnSinkhornNormalizeGetWorkspaceSize(t_in, eps, repeat, t_out, &ws_size, &executor);

    bool ok = (ret == 0);
    if (ok) {
        void* ws = nullptr;
        if (ws_size > 0) aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
        auto t0 = std::chrono::high_resolution_clock::now();
        aclnnSinkhornNormalize(ws, ws_size, executor, stream);
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        printf(" done (%ld us)\n", us);
        if (ws) aclrtFree(ws);
    } else {
        printf(" FAILED: GetWorkspaceSize=%d\n", ret);
    }

    // Read back
    std::vector<float> output(total, 0.0f);
    if (ok) {
        aclrtMemcpy(output.data(), data_size, d_out, data_size, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    // Cleanup ACL
    aclDestroyTensor(t_in);
    aclDestroyTensor(t_out);
    aclrtFree(d_in);
    aclrtFree(d_out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (!ok) return 1;

    // Precision metrics
    double max_err = 0, sum_err = 0;
    double max_rel_err = 0, sum_rel_err = 0;
    int err_count_1e4 = 0, err_count_1e3 = 0;

    for (int i = 0; i < total; i++) {
        double diff = fabs((double)output[i] - (double)golden[i]);
        sum_err += diff;
        if (diff > max_err) max_err = diff;
        double rel = diff / (fabs((double)golden[i]) + 1e-8);
        sum_rel_err += rel;
        if (rel > max_rel_err) max_rel_err = rel;
        if (diff > 1e-4) err_count_1e4++;
        if (diff > 1e-3) err_count_1e3++;
    }
    double mere = sum_rel_err / total;
    double mean_err = sum_err / total;

    const double mere_thr = 1.22e-4;
    const double mare_thr = 1.22e-3;
    bool mere_ok = mere < mere_thr;
    bool mare_ok = max_rel_err < mare_thr;

    printf("MERE: %.6e (%s), MARE: %.6e (%s)\n",
           mere, mere_ok ? "PASS" : "FAIL",
           max_rel_err, mare_ok ? "PASS" : "FAIL");
    printf("Max abs err: %.6e, Mean abs err: %.6e\n", max_err, mean_err);
    printf("Err > 1e-4: %d, Err > 1e-3: %d\n", err_count_1e4, err_count_1e3);

    // First matrix
    if (S >= 1) {
        printf("\nFirst 4x4 matrix (NPU / Golden):\n");
        for (int r = 0; r < 4; r++) {
            printf("  ");
            for (int c = 0; c < 4; c++)
                printf("%.6f ", output[r * 4 + c]);
            printf("| ");
            for (int c = 0; c < 4; c++)
                printf("%.6f ", golden[r * 4 + c]);
            printf("\n");
        }
    }

    return (mere_ok && mare_ok) ? 0 : 1;
}

int main(int argc, char** argv) {
    int S = 4, repeat = 10, seed = 42;
    float eps = 1e-6f;
    if (argc > 1) S = atoi(argv[1]);
    if (argc > 2) repeat = atoi(argv[2]);
    if (argc > 3) eps = atof(argv[3]);
    if (argc > 4) seed = atoi(argv[4]);
    return run_test(S, repeat, eps, seed);
}
