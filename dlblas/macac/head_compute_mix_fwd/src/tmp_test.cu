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

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    int B = 16;
    int N1 = 16384;
    int MHC = 4;
    float mhc_pre_eps = 1e-2f;
    int total = B * N1 * MHC;

    printf("Input shape: [%d, %d, %d]\n", B, N1, MHC);
    printf("Total elements: %d\n", total);

    // Allocate GPU memory
    float *input_mix = nullptr, *mhc_scale = nullptr, *mhc_base = nullptr;
    float *mhc_pre_eps_d = nullptr;
    float *output_ori = nullptr, *output_opt = nullptr;

    cudaMalloc((void**)&input_mix, sizeof(float) * total);
    cudaMalloc((void**)&mhc_scale, sizeof(float));
    cudaMalloc((void**)&mhc_base, sizeof(float) * MHC);
    cudaMalloc((void**)&mhc_pre_eps_d, sizeof(float));
    cudaMalloc((void**)&output_ori, sizeof(float) * total);
    cudaMalloc((void**)&output_opt, sizeof(float) * total);

    // CPU memory
    float *input_mix_cpu = (float*)malloc(sizeof(float) * total);
    float mhc_scale_cpu = 1.5f;
    float *mhc_base_cpu = (float*)malloc(sizeof(float) * MHC);
    float *output_ori_cpu = (float*)malloc(sizeof(float) * total);
    float *output_opt_cpu = (float*)malloc(sizeof(float) * total);

    // Initialize
    for (int i = 0; i < total; i++)
        input_mix_cpu[i] = (float)((i * 7) % 127) / 10.0f - 6.0f;
    for (int i = 0; i < MHC; i++)
        mhc_base_cpu[i] = (float)(i + 1) * 0.5f;

    cudaMemcpy(input_mix, input_mix_cpu, sizeof(float) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(mhc_scale, &mhc_scale_cpu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mhc_base, mhc_base_cpu, sizeof(float) * MHC, cudaMemcpyHostToDevice);
    cudaMemcpy(mhc_pre_eps_d, &mhc_pre_eps, sizeof(float), cudaMemcpyHostToDevice);

    CUDA_INIT();

    float ori_time = 0.f, opt_time = 0.f;

    // ori kernel
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori<float>(input_mix, mhc_scale, mhc_base, mhc_pre_eps_d,
                                       output_ori, total, MHC, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori<float>(input_mix, mhc_scale, mhc_base, mhc_pre_eps_d,
                                       output_ori, total, MHC, stream);
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

    // opt kernel
    if (exec_mode == 0 || exec_mode == 2) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt<float>(input_mix, mhc_scale, mhc_base, mhc_pre_eps_d,
                                       output_opt, total, MHC, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt<float>(input_mix, mhc_scale, mhc_base, mhc_pre_eps_d,
                                       output_opt, total, MHC, stream);
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

    // Precision check
    cudaDeviceSynchronize();
    bool is_right = true;
    if (exec_mode == 0) {
        cudaMemcpy(output_ori_cpu, output_ori, sizeof(float) * total, cudaMemcpyDeviceToHost);
        cudaMemcpy(output_opt_cpu, output_opt, sizeof(float) * total, cudaMemcpyDeviceToHost);
        is_right = check_result_custom<float>(output_ori_cpu, output_opt_cpu, total, 0.001f, "output");
    }

    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    float ratio = (ori_time > 0.f) ? (opt_time / ori_time) : 0.f;
    printf("<runtime_ratio> %f</runtime_ratio>\n", ratio);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // Cleanup
    cudaFree(input_mix); cudaFree(mhc_scale); cudaFree(mhc_base);
    cudaFree(mhc_pre_eps_d); cudaFree(output_ori); cudaFree(output_opt);
    free(input_mix_cpu); free(mhc_base_cpu);
    free(output_ori_cpu); free(output_opt_cpu);
    cudaStreamDestroy(stream);

    return 0;
}
