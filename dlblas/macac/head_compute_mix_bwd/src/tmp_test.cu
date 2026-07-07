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

    // Dimensions from the spec
    const int batch0 = 2;
    const int batch1 = 1024;
    const int mhc_mult = 4;
    const int total_elems = batch0 * batch1 * mhc_mult;

    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate inputs
    float *input_mix, *input_mix_cpu;
    float *mhc_scale, *mhc_scale_cpu;
    float *mhc_base, *mhc_base_cpu;
    float *grad_out, *grad_out_cpu;

    cudaMalloc((void**)&input_mix, sizeof(float) * total_elems);
    cudaMalloc((void**)&mhc_scale, sizeof(float) * 1);
    cudaMalloc((void**)&mhc_base, sizeof(float) * mhc_mult);
    cudaMalloc((void**)&grad_out, sizeof(float) * total_elems);

    input_mix_cpu = (float*)malloc(sizeof(float) * total_elems);
    mhc_scale_cpu = (float*)malloc(sizeof(float) * 1);
    mhc_base_cpu  = (float*)malloc(sizeof(float) * mhc_mult);
    grad_out_cpu  = (float*)malloc(sizeof(float) * total_elems);

    // Allocate outputs (two copies: ori and opt)
    float *grad_input_mix_ori, *grad_input_mix_opt;
    float *grad_mhc_scale_ori, *grad_mhc_scale_opt;
    float *grad_mhc_base_ori, *grad_mhc_base_opt;

    cudaMalloc((void**)&grad_input_mix_ori, sizeof(float) * total_elems);
    cudaMalloc((void**)&grad_input_mix_opt, sizeof(float) * total_elems);
    cudaMalloc((void**)&grad_mhc_scale_ori, sizeof(float) * 1);
    cudaMalloc((void**)&grad_mhc_scale_opt, sizeof(float) * 1);
    cudaMalloc((void**)&grad_mhc_base_ori, sizeof(float) * mhc_mult);
    cudaMalloc((void**)&grad_mhc_base_opt, sizeof(float) * mhc_mult);

    float *grad_input_mix_ori_cpu = (float*)malloc(sizeof(float) * total_elems);
    float *grad_input_mix_opt_cpu = (float*)malloc(sizeof(float) * total_elems);
    float *grad_mhc_scale_ori_cpu = (float*)malloc(sizeof(float) * 1);
    float *grad_mhc_scale_opt_cpu = (float*)malloc(sizeof(float) * 1);
    float *grad_mhc_base_ori_cpu  = (float*)malloc(sizeof(float) * mhc_mult);
    float *grad_mhc_base_opt_cpu  = (float*)malloc(sizeof(float) * mhc_mult);

    // Initialize inputs
    for (int i = 0; i < total_elems; i++) {
        input_mix_cpu[i] = (float)((i * 7) % 127) - 63.5f;
        grad_out_cpu[i] = (float)((i * 13) % 97) - 48.0f;
    }
    mhc_scale_cpu[0] = 1.5f;
    for (int i = 0; i < mhc_mult; i++) {
        mhc_base_cpu[i] = 0.5f + (float)i * 0.3f;
    }

    // Copy inputs to device
    cudaMemcpyAsync(input_mix, input_mix_cpu, sizeof(float) * total_elems, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mhc_scale, mhc_scale_cpu, sizeof(float) * 1, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mhc_base, mhc_base_cpu, sizeof(float) * mhc_mult, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(grad_out, grad_out_cpu, sizeof(float) * total_elems, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    float ori_time = 0.f, opt_time = 0.f;

    // Run ori kernel (exec_mode == 0 or 1)
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori<float>(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input_mix_ori, grad_mhc_scale_ori, grad_mhc_base_ori,
                batch0, batch1, mhc_mult, stream);
        cudaStreamSynchronize(stream);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori<float>(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input_mix_ori, grad_mhc_scale_ori, grad_mhc_base_ori,
                batch0, batch1, mhc_mult, stream);
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

    // Run opt kernel (exec_mode == 0 or 2)
    if (exec_mode == 0 || exec_mode == 2) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt<float>(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input_mix_opt, grad_mhc_scale_opt, grad_mhc_base_opt,
                batch0, batch1, mhc_mult, stream);
        cudaStreamSynchronize(stream);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt<float>(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input_mix_opt, grad_mhc_scale_opt, grad_mhc_base_opt,
                batch0, batch1, mhc_mult, stream);
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
    cudaStreamSynchronize(stream);
    bool is_right = true;
    if (exec_mode == 0) {
        cudaMemcpy(grad_input_mix_ori_cpu, grad_input_mix_ori, sizeof(float) * total_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_input_mix_opt_cpu, grad_input_mix_opt, sizeof(float) * total_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_mhc_scale_ori_cpu, grad_mhc_scale_ori, sizeof(float) * 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_mhc_scale_opt_cpu, grad_mhc_scale_opt, sizeof(float) * 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_mhc_base_ori_cpu, grad_mhc_base_ori, sizeof(float) * mhc_mult, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_mhc_base_opt_cpu, grad_mhc_base_opt, sizeof(float) * mhc_mult, cudaMemcpyDeviceToHost);

        bool ok1 = check_result_custom<float>(grad_input_mix_ori_cpu, grad_input_mix_opt_cpu, total_elems, 0.1f, "grad_input_mix");
        bool ok2 = check_result_custom<float>(grad_mhc_scale_ori_cpu, grad_mhc_scale_opt_cpu, 1, 0.1f, "grad_mhc_scale");
        bool ok3 = check_result_custom<float>(grad_mhc_base_ori_cpu, grad_mhc_base_opt_cpu, mhc_mult, 0.1f, "grad_mhc_base");
        is_right = ok1 && ok2 && ok3;
    }

    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f ? opt_time / ori_time : 0.f));
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // Cleanup
    free(input_mix_cpu); free(mhc_scale_cpu); free(mhc_base_cpu); free(grad_out_cpu);
    free(grad_input_mix_ori_cpu); free(grad_input_mix_opt_cpu);
    free(grad_mhc_scale_ori_cpu); free(grad_mhc_scale_opt_cpu);
    free(grad_mhc_base_ori_cpu); free(grad_mhc_base_opt_cpu);
    cudaFree(input_mix); cudaFree(mhc_scale); cudaFree(mhc_base); cudaFree(grad_out);
    cudaFree(grad_input_mix_ori); cudaFree(grad_input_mix_opt);
    cudaFree(grad_mhc_scale_ori); cudaFree(grad_mhc_scale_opt);
    cudaFree(grad_mhc_base_ori); cudaFree(grad_mhc_base_opt);

    cudaStreamDestroy(stream);
    return 0;
}
