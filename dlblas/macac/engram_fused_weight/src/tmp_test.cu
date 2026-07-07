#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

template<typename T>
bool check_result_fused(T* ref, T* test, int num_elements, float atol, const char* name) {
    int diff_nums = 0;
    for (int i = 0; i < num_elements; i++) {
        if (fabsf((float)test[i] - (float)ref[i]) > atol) {
            diff_nums++;
            if (diff_nums < 10)
                printf("[%s] Mismatch at %d: ref=%.6f, test=%.6f\n", name, i, (float)ref[i], (float)test[i]);
        }
    }
    if (diff_nums > 0) { printf("[%s] result is not right, %d mismatches\n", name, diff_nums); return false; }
    printf("[%s] result is right\n", name);
    return true;
}

__global__ void warm_up() {}

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    CUDA_INIT();

    // Operator dimensions (from engram_fused_weight.py)
    const int hc_mult = 4;
    const int hidden_size = 128;
    const int size = hc_mult * hidden_size;  // 512

    // ===== Allocate host memory =====
    __FLOAT16__* wh_data_cpu = (__FLOAT16__*)malloc(size * sizeof(__FLOAT16__));
    __FLOAT16__* we_data_cpu = (__FLOAT16__*)malloc(size * sizeof(__FLOAT16__));
    float* Y_ori_cpu = (float*)malloc(size * sizeof(float));
    float* Y_opt_cpu = (float*)malloc(size * sizeof(float));

    // ===== Initialize inputs with diverse values =====
    for (int i = 0; i < size; i++) {
        float v_wh = sinf((float)(i * 7) * 0.1f) * 0.5f;
        float v_we = cosf((float)(i * 13 + 3) * 0.1f) * 0.3f;
        wh_data_cpu[i] = (__FLOAT16__)v_wh;
        we_data_cpu[i] = (__FLOAT16__)v_we;
    }

    // ===== Allocate device memory =====
    __FLOAT16__ *wh_data, *we_data;
    float *Y_ori, *Y_opt;

    cudaMalloc((void**)&wh_data, size * sizeof(__FLOAT16__));
    cudaMalloc((void**)&we_data, size * sizeof(__FLOAT16__));
    cudaMalloc((void**)&Y_ori, size * sizeof(float));
    cudaMalloc((void**)&Y_opt, size * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(wh_data, wh_data_cpu, size * sizeof(__FLOAT16__), cudaMemcpyHostToDevice);
    cudaMemcpy(we_data, we_data_cpu, size * sizeof(__FLOAT16__), cudaMemcpyHostToDevice);

    float ori_time = 0.f, opt_time = 0.f;

    // ===== Run ori kernel (exec_mode 0 or 1) =====
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori(wh_data, we_data, Y_ori, size, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori(wh_data, we_data, Y_ori, size, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float elapsed = 0.f;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;
        }
        ori_time = total_time / test_count;
        printf("origin fprop average time: %f ms\n", ori_time);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // ===== Run opt kernel (exec_mode 0 or 2) =====
    if (exec_mode == 0 || exec_mode == 2) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt(wh_data, we_data, Y_opt, size, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt(wh_data, we_data, Y_opt, size, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float elapsed = 0.f;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;
        }
        opt_time = total_time / test_count;
        printf("opt fprop average time: %f ms\n", opt_time);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // ===== Precision check (only exec_mode == 0) =====
    cudaDeviceSynchronize();
    bool is_right = true;
    if (exec_mode == 0) {
        cudaMemcpy(Y_ori_cpu, Y_ori, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Y_opt_cpu, Y_opt, size * sizeof(float), cudaMemcpyDeviceToHost);

        float atol_f32 = 0.01f;
        is_right = check_result_fused<float>(Y_ori_cpu, Y_opt_cpu, size, atol_f32, "Y");
    }

    // ===== Output tags =====
    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f) ? (opt_time / ori_time) : 0.f);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // ===== Cleanup =====
    free(wh_data_cpu); free(we_data_cpu);
    free(Y_ori_cpu); free(Y_opt_cpu);

    cudaFree(wh_data); cudaFree(we_data);
    cudaFree(Y_ori); cudaFree(Y_opt);
    cudaStreamDestroy(stream);

    return 0;
}
