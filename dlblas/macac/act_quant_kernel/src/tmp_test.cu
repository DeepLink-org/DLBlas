#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

// Custom precision check for bf16 outputs
template<typename T>
bool check_result_custom(T* ref, T* test, int num_elements, float atol, const char* name) {
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

// Precision check for float outputs (scales)
bool check_result_float(float* ref, float* test, int num_elements, float atol, const char* name) {
    int diff_nums = 0;
    for (int i = 0; i < num_elements; i++) {
        if (fabsf(test[i] - ref[i]) > atol) {
            diff_nums++;
            if (diff_nums < 10)
                printf("[%s] Mismatch at %d: ref=%.6f, test=%.6f\n", name, i, ref[i], test[i]);
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

    // Problem dimensions (from get_inputs() in act_quant_kernel.py)
    int B = 7;
    int D = 512;
    int group_size = 512;
    float fp8_max = 448.0f;
    float fp8_min = -448.0f;
    int G = D / group_size;  // number of groups per row

    int x_numel = B * D;
    int xs_numel = B * G;

    typedef __FLOAT16__ T;

    // Allocate GPU memory for input
    T *x = nullptr;
    T *x_cpu = nullptr;
    cudaMalloc((void**)&x, sizeof(T) * x_numel);
    x_cpu = (T*)malloc(sizeof(T) * x_numel);

    // Allocate GPU memory for outputs (ori and opt versions)
    T *x_q_ori = nullptr;
    T *x_q_opt = nullptr;
    float *x_s_ori = nullptr;
    float *x_s_opt = nullptr;

    cudaMalloc((void**)&x_q_ori, sizeof(T) * x_numel);
    cudaMalloc((void**)&x_q_opt, sizeof(T) * x_numel);
    cudaMalloc((void**)&x_s_ori, sizeof(float) * xs_numel);
    cudaMalloc((void**)&x_s_opt, sizeof(float) * xs_numel);

    T *x_q_ori_cpu = (T*)malloc(sizeof(T) * x_numel);
    T *x_q_opt_cpu = (T*)malloc(sizeof(T) * x_numel);
    float *x_s_ori_cpu = (float*)malloc(sizeof(float) * xs_numel);
    float *x_s_opt_cpu = (float*)malloc(sizeof(float) * xs_numel);

    // Initialize input: bf16 random-ish values (matching torch.rand behavior)
    for (int i = 0; i < x_numel; i++) {
        // Generate values similar to torch.rand (uniform [0,1))
        // using a simple LCG-like pattern
        float val = (float)((i * 7 + 3) % 127) / 128.0f;
        x_cpu[i] = (T)val;
    }

    cudaMemcpy(x, x_cpu, sizeof(T) * x_numel, cudaMemcpyHostToDevice);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaSetDevice(0);

    float ori_time = 0.f, opt_time = 0.f;

    // exec_mode == 0 || exec_mode == 1: run ori kernel
    if (exec_mode == 0 || exec_mode == 1) {
        // Warmup
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori(x, x_q_ori, x_s_ori, B, D, group_size, fp8_max, fp8_min, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori(x, x_q_ori, x_s_ori, B, D, group_size, fp8_max, fp8_min, stream);
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

    // exec_mode == 0 || exec_mode == 2: run opt kernel
    if (exec_mode == 0 || exec_mode == 2) {
        // Warmup
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt(x, x_q_opt, x_s_opt, B, D, group_size, fp8_max, fp8_min, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt(x, x_q_opt, x_s_opt, B, D, group_size, fp8_max, fp8_min, stream);
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

    // Precision check (only exec_mode == 0)
    cudaDeviceSynchronize();
    bool is_right = true;
    if (exec_mode == 0) {
        // Copy results back to CPU
        cudaMemcpy(x_q_ori_cpu, x_q_ori, sizeof(T) * x_numel, cudaMemcpyDeviceToHost);
        cudaMemcpy(x_q_opt_cpu, x_q_opt, sizeof(T) * x_numel, cudaMemcpyDeviceToHost);
        cudaMemcpy(x_s_ori_cpu, x_s_ori, sizeof(float) * xs_numel, cudaMemcpyDeviceToHost);
        cudaMemcpy(x_s_opt_cpu, x_s_opt, sizeof(float) * xs_numel, cudaMemcpyDeviceToHost);

        bool x_q_ok = check_result_custom<T>(x_q_ori_cpu, x_q_opt_cpu, x_numel, 0.1f, "x_q");
        bool x_s_ok = check_result_float(x_s_ori_cpu, x_s_opt_cpu, xs_numel, 0.001f, "x_s");
        is_right = x_q_ok && x_s_ok;
    }

    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f) ? (opt_time / ori_time) : 1.0f);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(x); free(x_cpu);
    cudaFree(x_q_ori); cudaFree(x_q_opt);
    cudaFree(x_s_ori); cudaFree(x_s_opt);
    free(x_q_ori_cpu); free(x_q_opt_cpu);
    free(x_s_ori_cpu); free(x_s_opt_cpu);

    return 0;
}
