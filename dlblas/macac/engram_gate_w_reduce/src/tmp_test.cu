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

    const int B = 108;
    const int C = 4;
    const int H = 4096;
    const int total_outs = C * H;

    printf("<B>%d</B>\n", B);
    printf("<C>%d</C>\n", C);
    printf("<H>%d</H>\n", H);

    CUDA_INIT();

    // Allocate device memory
    float *grad_w_partial = nullptr;
    __FLOAT16__ *weight_hidden = nullptr;
    __FLOAT16__ *weight_embed = nullptr;
    float *grad_wh_ref = nullptr;
    float *grad_we_ref = nullptr;
    float *grad_wh_out_ori = nullptr;
    float *grad_we_out_ori = nullptr;
    float *grad_wh_out_opt = nullptr;
    float *grad_we_out_opt = nullptr;

    cudaMalloc((void**)&grad_w_partial, sizeof(float) * B * C * H);
    cudaMalloc((void**)&weight_hidden, sizeof(__FLOAT16__) * C * H);
    cudaMalloc((void**)&weight_embed, sizeof(__FLOAT16__) * C * H);
    cudaMalloc((void**)&grad_wh_ref, sizeof(float) * C * H);
    cudaMalloc((void**)&grad_we_ref, sizeof(float) * C * H);
    cudaMalloc((void**)&grad_wh_out_ori, sizeof(float) * C * H);
    cudaMalloc((void**)&grad_we_out_ori, sizeof(float) * C * H);
    cudaMalloc((void**)&grad_wh_out_opt, sizeof(float) * C * H);
    cudaMalloc((void**)&grad_we_out_opt, sizeof(float) * C * H);

    // Allocate host memory
    float *grad_w_partial_cpu = (float*)malloc(sizeof(float) * B * C * H);
    __FLOAT16__ *weight_hidden_cpu = (__FLOAT16__*)malloc(sizeof(__FLOAT16__) * C * H);
    __FLOAT16__ *weight_embed_cpu = (__FLOAT16__*)malloc(sizeof(__FLOAT16__) * C * H);
    float *grad_wh_ref_cpu = (float*)malloc(sizeof(float) * C * H);
    float *grad_we_ref_cpu = (float*)malloc(sizeof(float) * C * H);
    float *grad_wh_out_ori_cpu = (float*)malloc(sizeof(float) * C * H);
    float *grad_we_out_ori_cpu = (float*)malloc(sizeof(float) * C * H);
    float *grad_wh_out_opt_cpu = (float*)malloc(sizeof(float) * C * H);
    float *grad_we_out_opt_cpu = (float*)malloc(sizeof(float) * C * H);

    // Initialize inputs
    for (int i = 0; i < B * C * H; i++) {
        grad_w_partial_cpu[i] = (float)((i * 7) % 127) * 0.01f;
    }
    for (int i = 0; i < C * H; i++) {
        unsigned short hw = (unsigned short)((i * 13) % 127);
        unsigned short ew = (unsigned short)((i * 17) % 127);
        memcpy(&weight_hidden_cpu[i], &hw, sizeof(__FLOAT16__));
        memcpy(&weight_embed_cpu[i], &ew, sizeof(__FLOAT16__));
        grad_wh_ref_cpu[i] = (float)((i * 3) % 31) * 0.1f;
        grad_we_ref_cpu[i] = (float)((i * 5) % 29) * 0.1f;
    }

    // Copy to device
    cudaMemcpy(grad_w_partial, grad_w_partial_cpu, sizeof(float) * B * C * H, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_hidden, weight_hidden_cpu, sizeof(__FLOAT16__) * C * H, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_embed, weight_embed_cpu, sizeof(__FLOAT16__) * C * H, cudaMemcpyHostToDevice);
    cudaMemcpy(grad_wh_ref, grad_wh_ref_cpu, sizeof(float) * C * H, cudaMemcpyHostToDevice);
    cudaMemcpy(grad_we_ref, grad_we_ref_cpu, sizeof(float) * C * H, cudaMemcpyHostToDevice);

    float ori_time = 0.f, opt_time = 0.f;

    // exec_mode == 0 || exec_mode == 1: run ori kernel
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori(grad_w_partial, weight_hidden, weight_embed,
                                grad_wh_ref, grad_we_ref,
                                grad_wh_out_ori, grad_we_out_ori,
                                B, C, H, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_ori(grad_w_partial, weight_hidden, weight_embed,
                                grad_wh_ref, grad_we_ref,
                                grad_wh_out_ori, grad_we_out_ori,
                                B, C, H, stream);
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

    // exec_mode == 0 || exec_mode == 2: run opt kernel
    if (exec_mode == 0 || exec_mode == 2) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt(grad_w_partial, weight_hidden, weight_embed,
                                grad_wh_ref, grad_we_ref,
                                grad_wh_out_opt, grad_we_out_opt,
                                B, C, H, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_opt(grad_w_partial, weight_hidden, weight_embed,
                                grad_wh_ref, grad_we_ref,
                                grad_wh_out_opt, grad_we_out_opt,
                                B, C, H, stream);
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

    // Precision check (only exec_mode == 0)
    cudaDeviceSynchronize();
    bool is_right = true;
    if (exec_mode == 0) {
        cudaMemcpy(grad_wh_out_ori_cpu, grad_wh_out_ori, sizeof(float) * C * H, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_wh_out_opt_cpu, grad_wh_out_opt, sizeof(float) * C * H, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_we_out_ori_cpu, grad_we_out_ori, sizeof(float) * C * H, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_we_out_opt_cpu, grad_we_out_opt, sizeof(float) * C * H, cudaMemcpyDeviceToHost);

        bool wh_ok = check_result_custom<float>(grad_wh_out_ori_cpu, grad_wh_out_opt_cpu, C * H, 0.1f, "grad_wh_out");
        bool we_ok = check_result_custom<float>(grad_we_out_ori_cpu, grad_we_out_opt_cpu, C * H, 0.1f, "grad_we_out");
        is_right = wh_ok && we_ok;
    }

    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", ori_time > 0.f ? opt_time / ori_time : 0.f);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // Cleanup
    cudaFree(grad_w_partial); cudaFree(weight_hidden); cudaFree(weight_embed);
    cudaFree(grad_wh_ref); cudaFree(grad_we_ref);
    cudaFree(grad_wh_out_ori); cudaFree(grad_we_out_ori);
    cudaFree(grad_wh_out_opt); cudaFree(grad_we_out_opt);
    free(grad_w_partial_cpu); free(weight_hidden_cpu); free(weight_embed_cpu);
    free(grad_wh_ref_cpu); free(grad_we_ref_cpu);
    free(grad_wh_out_ori_cpu); free(grad_we_out_ori_cpu);
    free(grad_wh_out_opt_cpu); free(grad_we_out_opt_cpu);

    return 0;
}
