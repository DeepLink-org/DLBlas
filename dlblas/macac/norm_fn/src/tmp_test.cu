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
ResultStruct test_tmp(
    int num_rows, int num_mixes, int rms_group_size, float eps,
    int warm_up_count, int test_count, int exec_mode
) {
    ResultStruct result;
    CUDA_INIT();

    int residual_elems = num_rows * rms_group_size;
    int mhc_fn_elems   = num_mixes * rms_group_size;
    int output_elems   = num_rows * num_mixes;

    T *residual     = nullptr;
    T *mhc_fn       = nullptr;
    T *output_ori   = nullptr;
    T *output_opt   = nullptr;
    T *residual_cpu = nullptr;
    T *mhc_fn_cpu   = nullptr;
    T *output_ori_cpu   = nullptr;
    T *output_opt_cpu   = nullptr;

    // Allocate CPU memory
    residual_cpu     = (T *)malloc(sizeof(T) * residual_elems);
    mhc_fn_cpu       = (T *)malloc(sizeof(T) * mhc_fn_elems);
    output_ori_cpu   = (T *)malloc(sizeof(T) * output_elems);
    output_opt_cpu   = (T *)malloc(sizeof(T) * output_elems);

    // Allocate GPU memory
    cudaMalloc((void **)&residual,   sizeof(T) * residual_elems);
    cudaMalloc((void **)&mhc_fn,     sizeof(T) * mhc_fn_elems);
    cudaMalloc((void **)&output_ori, sizeof(T) * output_elems);
    cudaMalloc((void **)&output_opt, sizeof(T) * output_elems);

    // Initialize input data
    for (int i = 0; i < residual_elems; i++) {
        residual_cpu[i] = (T)((i * 7) % 127);
    }
    for (int i = 0; i < mhc_fn_elems; i++) {
        mhc_fn_cpu[i] = (T)(((i * 13) % 127) * 1e-4f);
    }

    // Copy inputs to GPU
    cudaMemcpy(residual, residual_cpu, sizeof(T) * residual_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(mhc_fn,   mhc_fn_cpu,   sizeof(T) * mhc_fn_elems,   cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    // exec_mode == 0 or 1: run ori kernel
    if (exec_mode == 0 || exec_mode == 1) {
        float total_time = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_ori(residual, mhc_fn, output_ori,
                                num_rows, num_mixes, rms_group_size, eps, stream);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_ori(residual, mhc_fn, output_ori,
                                num_rows, num_mixes, rms_group_size, eps, stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsed_time = 0.0f;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            total_time += elapsed_time;
        }
        result.ori_time = total_time / (float)test_count;
        printf("origin fprop average time: %f ms\n", result.ori_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // exec_mode == 0 or 2: run opt kernel
    if (exec_mode == 0 || exec_mode == 2) {
        float total_time = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_opt(residual, mhc_fn, output_opt,
                                num_rows, num_mixes, rms_group_size, eps, stream);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_opt(residual, mhc_fn, output_opt,
                                num_rows, num_mixes, rms_group_size, eps, stream);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsed_time = 0.0f;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            total_time += elapsed_time;
        }
        result.opt_time = total_time / (float)test_count;
        printf("opt   fprop average time: %f ms\n", result.opt_time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(output_ori_cpu, output_ori, sizeof(T) * output_elems, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_opt_cpu, output_opt, sizeof(T) * output_elems, cudaMemcpyDeviceToHost);

    // Precision check (only exec_mode 0)
    if (exec_mode == 0) {
        result.time_rate = result.opt_time / result.ori_time;
        result.result_is_right = checkresult<T>(output_ori_cpu, output_opt_cpu, output_elems);
    } else {
        result.time_rate = (result.ori_time > 0 && result.opt_time > 0)
                               ? result.opt_time / result.ori_time
                               : 0.0f;
        result.result_is_right = true;
    }

    // Free memory
    free(residual_cpu);
    free(mhc_fn_cpu);
    free(output_ori_cpu);
    free(output_opt_cpu);
    cudaFree(residual);
    cudaFree(mhc_fn);
    cudaFree(output_ori);
    cudaFree(output_opt);

    return result;
}

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    // norm_fn parameters (from test data specification)
    int num_rows      = 13;
    int num_mixes     = 24;
    int rms_group_size = 5120;
    float eps         = 1e-6f;

    ResultStruct result1 = test_tmp<float>(
        num_rows, num_mixes, rms_group_size, eps,
        warm_up_count, test_count, exec_mode
    );

    printf("<time_before_opt>%f ms</time_before_opt>\n", result1.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", result1.opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", result1.opt_time / result1.ori_time);
    if (result1.result_is_right) {
        printf("<precision>True</precision>\n");
    } else {
        printf("<precision>False</precision>\n");
    }
}
