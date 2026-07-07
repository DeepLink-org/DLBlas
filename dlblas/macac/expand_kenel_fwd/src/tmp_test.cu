#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

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

__global__ void warm_up() {}

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    // Dimensions from the torch spec: (batch=1, seq_len=1024, mhc_mult=4, hidden_size=1280)
    int batch_size = 1;
    int seq_len = 1024;
    int mhc_mult = 4;
    int hidden_size = 1280;

    int input_elems = batch_size * seq_len * hidden_size;
    int output_elems = batch_size * seq_len * mhc_mult * hidden_size;

    printf("Input shape: [%d, %d, %d] -> Output shape: [%d, %d, %d, %d]\n",
           batch_size, seq_len, hidden_size,
           batch_size, seq_len, mhc_mult, hidden_size);
    printf("Input elements: %d, Output elements: %d\n", input_elems, output_elems);

    // Allocate memory
    float* input = nullptr;
    float* output_ori = nullptr;
    float* output_opt = nullptr;
    float* input_cpu = nullptr;
    float* output_ori_cpu = nullptr;
    float* output_opt_cpu = nullptr;

    cudaMalloc((void**)&input, sizeof(float) * input_elems);
    cudaMalloc((void**)&output_ori, sizeof(float) * output_elems);
    cudaMalloc((void**)&output_opt, sizeof(float) * output_elems);
    input_cpu = (float*)malloc(sizeof(float) * input_elems);
    output_ori_cpu = (float*)malloc(sizeof(float) * output_elems);
    output_opt_cpu = (float*)malloc(sizeof(float) * output_elems);

    // Initialize input
    for (int i = 0; i < input_elems; i++) {
        input_cpu[i] = (float)((i * 7) % 127);
    }
    cudaMemcpy(input, input_cpu, sizeof(float) * input_elems, cudaMemcpyHostToDevice);

    CUDA_INIT();

    float ori_time = 0.f, opt_time = 0.f;

    // exec_mode == 0 (both) or exec_mode == 1 (ori only)
    if (exec_mode == 0 || exec_mode == 1) {
        // Warm up
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori<float>(input, output_ori, batch_size, seq_len, mhc_mult, hidden_size, stream);
        cudaDeviceSynchronize();

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori<float>(input, output_ori, batch_size, seq_len, mhc_mult, hidden_size, stream);
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

    // exec_mode == 0 (both) or exec_mode == 2 (opt only)
    if (exec_mode == 0 || exec_mode == 2) {
        // Warm up
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt<float>(input, output_opt, batch_size, seq_len, mhc_mult, hidden_size, stream);
        cudaDeviceSynchronize();

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt<float>(input, output_opt, batch_size, seq_len, mhc_mult, hidden_size, stream);
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

    // Correctness check (only exec_mode == 0)
    cudaDeviceSynchronize();
    bool is_right = true;
    if (exec_mode == 0) {
        cudaMemcpy(output_ori_cpu, output_ori, sizeof(float) * output_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(output_opt_cpu, output_opt, sizeof(float) * output_elems, cudaMemcpyDeviceToHost);
        is_right = check_result_custom<float>(output_ori_cpu, output_opt_cpu, output_elems, 0.1f, "output");
    }

    // Output tags
    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f ? opt_time / ori_time : 0.f));
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
