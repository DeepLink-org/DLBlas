#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

struct ResultStruct {
    float ori_time;
    float opt_time;
    float time_rate;
    bool result_is_right;
};

__global__ void warm_up()
{
}

template <typename T>
ResultStruct test_tmp(std::vector<int> input_size, std::vector<int> output_size,
                      int warm_up_count, int test_count, int exec_mode) {
    ResultStruct result;
    CUDA_INIT();

    int in_batch    = input_size[0];
    int in_height   = input_size[1];
    int in_channels = input_size[2];
    int in_width    = input_size[3];

    int out_batch    = output_size[0];
    int out_height   = output_size[1];
    int out_channels = output_size[2];
    int out_width    = output_size[3];

    int in_elems  = in_batch * in_height * in_channels * in_width;
    int out_elems = out_batch * out_height * out_channels * out_width;

    T *input = nullptr;
    T *output = nullptr;
    T *output_target = nullptr;
    T *input_cpu = nullptr;
    T *output_cpu = nullptr;
    T *output_target_cpu = nullptr;

    // Allocate CPU memory
    input_cpu = (T *)malloc(sizeof(T) * in_elems);
    output_cpu = (T *)malloc(sizeof(T) * out_elems);
    output_target_cpu = (T *)malloc(sizeof(T) * out_elems);

    // Allocate GPU memory
    cudaMalloc((void **)&input, sizeof(T) * in_elems);
    cudaMalloc((void **)&output, sizeof(T) * out_elems);
    cudaMalloc((void **)&output_target, sizeof(T) * out_elems);

    // Initialize output buffers with different values for verification
    memset(output_cpu, 1, sizeof(T) * out_elems);
    memset(output_target_cpu, 2, sizeof(T) * out_elems);
    cudaMemset(output, 1, sizeof(T) * out_elems);
    cudaMemset(output_target, 2, sizeof(T) * out_elems);

    // Initialize input with deterministic pattern
    for (int i = 0; i < in_elems; i++) {
        input_cpu[i] = (T)(((i * 7) % 127) * 0.01f - 0.635f);
    }
    // Copy input to GPU
    cudaMemcpy(input, input_cpu, sizeof(T) * in_elems, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    // exec_mode == 0: both ori and opt (precision comparison)
    // exec_mode == 1: only ori
    // exec_mode == 2: only opt
    if (exec_mode == 0 || exec_mode == 1) {
        float total_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_ori(input, output,
                in_batch, in_height, in_channels, in_width,
                out_batch, out_height, out_channels, out_width,
                in_elems, out_elems, stream);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_ori(input, output,
                in_batch, in_height, in_channels, in_width,
                out_batch, out_height, out_channels, out_width,
                in_elems, out_elems, stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed_time = 0.0f;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            total_time += elapsed_time;
        }
        result.ori_time = total_time / (float)(test_count);
        printf("origin fprop average time: %f ms\n", result.ori_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    if (exec_mode == 0 || exec_mode == 2) {
        float total_time = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_opt(input, output_target,
                in_batch, in_height, in_channels, in_width,
                out_batch, out_height, out_channels, out_width,
                in_elems, out_elems, stream);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_opt(input, output_target,
                in_batch, in_height, in_channels, in_width,
                out_batch, out_height, out_channels, out_width,
                in_elems, out_elems, stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed_time = 0.0f;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            total_time += elapsed_time;
        }
        result.opt_time = total_time / (float)(test_count);
        printf("opt   fprop average time: %f ms\n", result.opt_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaDeviceSynchronize();

    // Copy results from GPU to CPU
    cudaMemcpy(output_cpu, output, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_target_cpu, output_target, sizeof(T) * out_elems, cudaMemcpyDeviceToHost);

    // Precision comparison
    if (exec_mode == 0) {
        result.time_rate = result.opt_time / result.ori_time;
        result.result_is_right = checkresult<T>(output_cpu, output_target_cpu, out_elems);
    } else {
        result.time_rate = (result.ori_time > 0 && result.opt_time > 0) ?
                           result.opt_time / result.ori_time : 0.0f;
        result.result_is_right = true;
    }

    // Free memory
    free(input_cpu);
    free(output_cpu);
    free(output_target_cpu);
    cudaFree(input);
    cudaFree(output);
    cudaFree(output_target);

    return result;
}

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    // MTPBlock HC kernel: input [B, S, HC, D] -> output [B, S, D]
    // Default test dimensions: B=1, S=8, HC=4, D=512
    ResultStruct result1 = test_tmp<float>(
        {1, 8, 4, 512},      // input:  [B, S, HC, D]
        {1, 8, 512, 1},      // output: [B, S, D, 1]
        warm_up_count, test_count, exec_mode);

    printf("<time_before_opt>%f ms</time_before_opt>\n", result1.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", result1.opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", result1.opt_time / result1.ori_time);
    if (result1.result_is_right) {
        printf("<precision>True</precision>\n");
    } else {
        printf("<precision>False</precision>\n");
    }
}
