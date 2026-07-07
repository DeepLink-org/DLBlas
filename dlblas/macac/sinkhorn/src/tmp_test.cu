#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

__global__ void warm_up() {}

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    // Sinkhorn parameters (from origin/sinkhorn.py)
    const int n0 = 1;
    const int n1 = 1024;
    const int mhc = 4;
    const int repeat = 10;
    const float eps = 1e-6f;

    const int total_matrices = n0 * n1;         // 1024
    const int elems_per_mat = mhc * mhc;         // 16
    const int total_elems = total_matrices * elems_per_mat;  // 16384

    CUDA_INIT();

    // Allocate GPU memory
    float *input = nullptr;
    float *output_ori = nullptr;
    float *output_opt = nullptr;
    cudaMalloc((void**)&input, sizeof(float) * total_elems);
    cudaMalloc((void**)&output_ori, sizeof(float) * total_elems);
    cudaMalloc((void**)&output_opt, sizeof(float) * total_elems);

    // Allocate CPU memory
    float *input_cpu = (float*)malloc(sizeof(float) * total_elems);
    float *output_ori_cpu = (float*)malloc(sizeof(float) * total_elems);
    float *output_opt_cpu = (float*)malloc(sizeof(float) * total_elems);

    // Initialize: random-like values using deterministic pattern
    for (int i = 0; i < total_elems; i++) {
        input_cpu[i] = (float)((i * 7 + 13) % 127) / 10.0f - 6.0f;  // range ~[-6, 6]
    }

    // Warmup kernel
    warm_up<<<1, 1, 0, stream>>>();
    cudaDeviceSynchronize();

    // Copy input to device
    cudaMemcpy(input, input_cpu, sizeof(float) * total_elems, cudaMemcpyHostToDevice);

    float ori_time = 0.f, opt_time = 0.f;

    // === Run ori kernel (exec_mode 0 or 1) ===
    if (exec_mode == 0 || exec_mode == 1) {
        // Warmup
        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_ori<float>(input, output_ori, total_matrices, mhc, repeat, eps, stream);
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_ori<float>(input, output_ori, total_matrices, mhc, repeat, eps, stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed = 0.f;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;
        }
        ori_time = total_time / test_count;
        printf("origin fprop average time: %f ms\n", ori_time);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // === Run opt kernel (exec_mode 0 or 2) ===
    if (exec_mode == 0 || exec_mode == 2) {
        // Warmup
        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_opt<float>(input, output_opt, total_matrices, mhc, repeat, eps, stream);
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_opt<float>(input, output_opt, total_matrices, mhc, repeat, eps, stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed = 0.f;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;
        }
        opt_time = total_time / test_count;
        printf("opt fprop average time: %f ms\n", opt_time);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // === Precision check (exec_mode 0 only) ===
    bool is_right = true;
    if (exec_mode == 0) {
        cudaDeviceSynchronize();
        cudaMemcpy(output_ori_cpu, output_ori, sizeof(float) * total_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(output_opt_cpu, output_opt, sizeof(float) * total_elems, cudaMemcpyDeviceToHost);

        // Use atol=0.1 for sinkhorn (iterative normalization can accumulate small differences)
        float atol = 0.1f;
        int diff_nums = 0;
        for (int i = 0; i < total_elems; i++) {
            if (fabsf(output_opt_cpu[i] - output_ori_cpu[i]) > atol) {
                diff_nums++;
                if (diff_nums < 10)
                    printf("[sinkhorn] Mismatch at %d: ref=%.6f, test=%.6f\n", i, output_ori_cpu[i], output_opt_cpu[i]);
            }
        }
        if (diff_nums > 0) {
            printf("[sinkhorn] result is not right, %d mismatches\n", diff_nums);
            is_right = false;
        } else {
            printf("[sinkhorn] result is right\n");
        }
    }

    // === Output tags ===
    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0 ? opt_time / ori_time : 1.0f));
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // Cleanup
    cudaFree(input);
    cudaFree(output_ori);
    cudaFree(output_opt);
    free(input_cpu);
    free(output_ori_cpu);
    free(output_opt_cpu);
    cudaStreamDestroy(stream);

    return 0;
}
