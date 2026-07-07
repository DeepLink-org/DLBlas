#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

// Reference CPU implementation matching Python:
// term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
// output = bf16(x.float().unsqueeze(-2) * post_layer_mix + term2)

void cpu_mhc_post_ref(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_layer_mix,
    const float* comb_res_mix,
    float* output,  // float for comparison
    int n0, int n1, int h, int mhc_mult
) {
    int total_bs = n0 * n1;
    for (int bs = 0; bs < total_bs; bs++) {
        const __nv_bfloat16* x_bs = x + bs * h;
        const __nv_bfloat16* residual_bs = residual + bs * mhc_mult * h;
        const float* plm_bs = post_layer_mix + bs * mhc_mult;
        const float* crm_bs = comb_res_mix + bs * mhc_mult * mhc_mult;
        float* output_bs = output + bs * mhc_mult * h;

        for (int hi = 0; hi < h; hi++) {
            float x_val = __bfloat162float(x_bs[hi]);

            float res[4];
            for (int k = 0; k < mhc_mult; k++) {
                res[k] = __bfloat162float(residual_bs[k * h + hi]);
            }

            for (int m = 0; m < mhc_mult; m++) {
                float term2 = 0.0f;
                for (int k = 0; k < mhc_mult; k++) {
                    term2 += crm_bs[m * mhc_mult + k] * res[k];
                }
                output_bs[m * h + hi] = x_val * plm_bs[m] + term2;
            }
        }
    }
}

template<typename T>
bool check_result_custom(T* ref, T* test, int num_elements, float atol, const char* name) {
    int diff_nums = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < num_elements; i++) {
        float diff = fabsf((float)test[i] - (float)ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > atol) {
            diff_nums++;
            if (diff_nums < 10)
                printf("[%s] Mismatch at %d: ref=%.6f, test=%.6f, diff=%.6f\n", name, i, (float)ref[i], (float)test[i], diff);
        }
    }
    printf("[%s] max_diff=%.6f, atol=%.6f\n", name, max_diff, atol);
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

    // Problem dimensions matching Python spec
    const int n0 = 2;
    const int n1 = 4096;
    const int h = 1280;
    const int mhc_mult = 4;
    const int total_bs = n0 * n1;          // 8192
    const int total_x = total_bs * h;       // 10,485,760
    const int total_residual = total_bs * mhc_mult * h;  // 41,943,040
    const int total_plm = total_bs * mhc_mult;           // 32,768
    const int total_crm = total_bs * mhc_mult * mhc_mult; // 131,072
    const int total_out = total_bs * mhc_mult * h;        // 41,943,040

    printf("<n0>%d</n0>\n", n0);
    printf("<n1>%d</n1>\n", n1);
    printf("<h>%d</h>\n", h);
    printf("<mhc_mult>%d</mhc_mult>\n", mhc_mult);
    printf("<total_elements>%d</total_elements>\n", total_out);

    CUDA_INIT();

    // Allocate GPU memory
    __nv_bfloat16 *x = nullptr, *residual = nullptr;
    float *post_layer_mix = nullptr, *comb_res_mix = nullptr;
    __nv_bfloat16 *out_ori = nullptr, *out_opt = nullptr;

    cudaMalloc((void**)&x, sizeof(__nv_bfloat16) * total_x);
    cudaMalloc((void**)&residual, sizeof(__nv_bfloat16) * total_residual);
    cudaMalloc((void**)&post_layer_mix, sizeof(float) * total_plm);
    cudaMalloc((void**)&comb_res_mix, sizeof(float) * total_crm);
    cudaMalloc((void**)&out_ori, sizeof(__nv_bfloat16) * total_out);
    cudaMalloc((void**)&out_opt, sizeof(__nv_bfloat16) * total_out);

    // Allocate CPU memory
    __nv_bfloat16 *x_cpu = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * total_x);
    __nv_bfloat16 *residual_cpu = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * total_residual);
    float *plm_cpu = (float*)malloc(sizeof(float) * total_plm);
    float *crm_cpu = (float*)malloc(sizeof(float) * total_crm);
    __nv_bfloat16 *out_ori_cpu = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * total_out);
    __nv_bfloat16 *out_opt_cpu = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * total_out);
    float *out_ref_cpu = (float*)malloc(sizeof(float) * total_out);
    float *out_ori_float = (float*)malloc(sizeof(float) * total_out);
    float *out_opt_float = (float*)malloc(sizeof(float) * total_out);

    // Initialize with deterministic data (hash-based for reproducibility)
    for (int i = 0; i < total_x; i++) {
        float val = sinf((float)(i * 7 + 13) * 0.001f) * 0.5f;
        x_cpu[i] = __float2bfloat16(val);
    }
    for (int i = 0; i < total_residual; i++) {
        float val = cosf((float)(i * 11 + 3) * 0.001f) * 0.5f;
        residual_cpu[i] = __float2bfloat16(val);
    }
    for (int i = 0; i < total_plm; i++) {
        plm_cpu[i] = sinf((float)(i * 5 + 17) * 0.01f) * 0.5f + 0.5f;
    }
    for (int i = 0; i < total_crm; i++) {
        crm_cpu[i] = cosf((float)(i * 3 + 23) * 0.01f) * 0.3f;
    }

    // Compute CPU reference
    cpu_mhc_post_ref(x_cpu, residual_cpu, plm_cpu, crm_cpu, out_ref_cpu, n0, n1, h, mhc_mult);

    // Copy inputs to GPU
    cudaMemcpy(x, x_cpu, sizeof(__nv_bfloat16) * total_x, cudaMemcpyHostToDevice);
    cudaMemcpy(residual, residual_cpu, sizeof(__nv_bfloat16) * total_residual, cudaMemcpyHostToDevice);
    cudaMemcpy(post_layer_mix, plm_cpu, sizeof(float) * total_plm, cudaMemcpyHostToDevice);
    cudaMemcpy(comb_res_mix, crm_cpu, sizeof(float) * total_crm, cudaMemcpyHostToDevice);

    float ori_time = 0.f, opt_time = 0.f;

    // Run original kernel
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori(x, residual, post_layer_mix, comb_res_mix, out_ori,
                                n0, n1, h, mhc_mult, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_ori(x, residual, post_layer_mix, comb_res_mix, out_ori,
                                n0, n1, h, mhc_mult, stream);
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

    // Run optimized kernel
    if (exec_mode == 0 || exec_mode == 2) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_opt(x, residual, post_layer_mix, comb_res_mix, out_opt,
                                n0, n1, h, mhc_mult, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, 0);
            test_tmp_kernel_opt(x, residual, post_layer_mix, comb_res_mix, out_opt,
                                n0, n1, h, mhc_mult, stream);
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

    cudaDeviceSynchronize();

    // Correctness check
    bool is_right = true;
    if (exec_mode == 0) {
        cudaMemcpy(out_ori_cpu, out_ori, sizeof(__nv_bfloat16) * total_out, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_opt_cpu, out_opt, sizeof(__nv_bfloat16) * total_out, cudaMemcpyDeviceToHost);

        // Convert to float for comparison
        for (int i = 0; i < total_out; i++) {
            out_ori_float[i] = __bfloat162float(out_ori_cpu[i]);
            out_opt_float[i] = __bfloat162float(out_opt_cpu[i]);
        }

        // Compare opt vs ori (should match exactly on same hardware)
        bool ori_vs_opt = check_result_custom<float>(out_ori_float, out_opt_float, total_out, 0.01f, "opt_vs_ori");

        // Compare ori vs CPU reference
        bool ori_vs_ref = check_result_custom<float>(out_ref_cpu, out_ori_float, total_out, 0.02f, "ori_vs_ref");

        is_right = ori_vs_opt && ori_vs_ref;
    }

    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f) ? (opt_time / ori_time) : 0.f);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // Cleanup
    free(x_cpu); free(residual_cpu); free(plm_cpu); free(crm_cpu);
    free(out_ori_cpu); free(out_opt_cpu); free(out_ref_cpu);
    free(out_ori_float); free(out_opt_float);
    cudaFree(x); cudaFree(residual); cudaFree(post_layer_mix);
    cudaFree(comb_res_mix); cudaFree(out_ori); cudaFree(out_opt);
    cudaStreamDestroy(stream);
    return 0;
}
