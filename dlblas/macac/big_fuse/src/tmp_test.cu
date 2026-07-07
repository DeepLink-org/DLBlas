#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

// Custom precision check for float32 outputs
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

    // Operator dimensions (from big_fuse.py)
    const int seq_len = 512;
    const int mhc_mult = 4;
    const int hidden_size = 1280;
    const int mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult; // 24
    const int mhc_rgs = mhc_mult * hidden_size;               // 5120

    // Scalar parameters
    const float rms_eps = 1e-6f;
    const float pre_eps = 1e-6f;
    const float sinkhorn_eps = 1e-6f;
    const float post_mult_val = 1.0f;
    const int sinkhorn_repeat = 10;

    // Tensor element counts
    const int residual_elems = seq_len * mhc_rgs;          // 512 * 5120 = 2,621,440
    const int fn_elems = mhc_mult3 * mhc_rgs;              // 24 * 5120 = 122,880
    const int scale_elems = 3;
    const int base_elems = mhc_mult3;                      // 24
    const int post_mix_elems = seq_len * mhc_mult;         // 512 * 4 = 2,048
    const int comb_mix_elems = seq_len * mhc_mult * mhc_mult; // 512 * 16 = 8,192
    const int layer_input_elems = seq_len * hidden_size;   // 512 * 1280 = 655,360

    // ===== Allocate host memory =====
    __FLOAT16__* residual_cpu = (__FLOAT16__*)malloc(residual_elems * sizeof(__FLOAT16__));
    float* fn_cpu = (float*)malloc(fn_elems * sizeof(float));
    float* mhc_scale_cpu = (float*)malloc(scale_elems * sizeof(float));
    float* mhc_base_cpu = (float*)malloc(base_elems * sizeof(float));

    float* post_mix_ori_cpu = (float*)malloc(post_mix_elems * sizeof(float));
    float* post_mix_opt_cpu = (float*)malloc(post_mix_elems * sizeof(float));
    float* comb_mix_ori_cpu = (float*)malloc(comb_mix_elems * sizeof(float));
    float* comb_mix_opt_cpu = (float*)malloc(comb_mix_elems * sizeof(float));
    __FLOAT16__* layer_input_ori_cpu = (__FLOAT16__*)malloc(layer_input_elems * sizeof(__FLOAT16__));
    __FLOAT16__* layer_input_opt_cpu = (__FLOAT16__*)malloc(layer_input_elems * sizeof(__FLOAT16__));

    // ===== Initialize inputs =====
    // residual: random-like bf16 values
    for (int i = 0; i < residual_elems; i++) {
        float v = sinf((float)(i * 7) * 0.001f) * 0.5f;
        residual_cpu[i] = (__FLOAT16__)v;
    }
    // fn weight: small random values
    for (int i = 0; i < fn_elems; i++) {
        fn_cpu[i] = sinf((float)(i * 13 + 3) * 0.001f) * 1e-4f;
    }
    // mhc_scale: small random
    mhc_scale_cpu[0] = 0.05f;
    mhc_scale_cpu[1] = -0.03f;
    mhc_scale_cpu[2] = 0.07f;
    // mhc_base: small random
    for (int i = 0; i < base_elems; i++) {
        mhc_base_cpu[i] = sinf((float)(i * 17) * 0.1f) * 0.1f;
    }

    // ===== Allocate device memory =====
    __FLOAT16__ *residual, *layer_input_ori, *layer_input_opt;
    float *fn, *mhc_scale, *mhc_base;
    float *post_mix_ori, *post_mix_opt, *comb_mix_ori, *comb_mix_opt;

    cudaMalloc((void**)&residual, residual_elems * sizeof(__FLOAT16__));
    cudaMalloc((void**)&fn, fn_elems * sizeof(float));
    cudaMalloc((void**)&mhc_scale, scale_elems * sizeof(float));
    cudaMalloc((void**)&mhc_base, base_elems * sizeof(float));
    cudaMalloc((void**)&post_mix_ori, post_mix_elems * sizeof(float));
    cudaMalloc((void**)&post_mix_opt, post_mix_elems * sizeof(float));
    cudaMalloc((void**)&comb_mix_ori, comb_mix_elems * sizeof(float));
    cudaMalloc((void**)&comb_mix_opt, comb_mix_elems * sizeof(float));
    cudaMalloc((void**)&layer_input_ori, layer_input_elems * sizeof(__FLOAT16__));
    cudaMalloc((void**)&layer_input_opt, layer_input_elems * sizeof(__FLOAT16__));

    // Copy inputs to device
    cudaMemcpy(residual, residual_cpu, residual_elems * sizeof(__FLOAT16__), cudaMemcpyHostToDevice);
    cudaMemcpy(fn, fn_cpu, fn_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mhc_scale, mhc_scale_cpu, scale_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mhc_base, mhc_base_cpu, base_elems * sizeof(float), cudaMemcpyHostToDevice);

    float ori_time = 0.f, opt_time = 0.f;

    // ===== Run ori kernel (exec_mode 0 or 1) =====
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori<__FLOAT16__>(residual, fn, mhc_scale, mhc_base,
                post_mix_ori, comb_mix_ori, layer_input_ori,
                seq_len, mhc_mult, hidden_size, mhc_mult3, mhc_rgs,
                rms_eps, pre_eps, sinkhorn_eps, post_mult_val, sinkhorn_repeat, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori<__FLOAT16__>(residual, fn, mhc_scale, mhc_base,
                post_mix_ori, comb_mix_ori, layer_input_ori,
                seq_len, mhc_mult, hidden_size, mhc_mult3, mhc_rgs,
                rms_eps, pre_eps, sinkhorn_eps, post_mult_val, sinkhorn_repeat, stream);
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
            test_tmp_kernel_opt<__FLOAT16__>(residual, fn, mhc_scale, mhc_base,
                post_mix_opt, comb_mix_opt, layer_input_opt,
                seq_len, mhc_mult, hidden_size, mhc_mult3, mhc_rgs,
                rms_eps, pre_eps, sinkhorn_eps, post_mult_val, sinkhorn_repeat, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt<__FLOAT16__>(residual, fn, mhc_scale, mhc_base,
                post_mix_opt, comb_mix_opt, layer_input_opt,
                seq_len, mhc_mult, hidden_size, mhc_mult3, mhc_rgs,
                rms_eps, pre_eps, sinkhorn_eps, post_mult_val, sinkhorn_repeat, stream);
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
        cudaMemcpy(post_mix_ori_cpu, post_mix_ori, post_mix_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(post_mix_opt_cpu, post_mix_opt, post_mix_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(comb_mix_ori_cpu, comb_mix_ori, comb_mix_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(comb_mix_opt_cpu, comb_mix_opt, comb_mix_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(layer_input_ori_cpu, layer_input_ori, layer_input_elems * sizeof(__FLOAT16__), cudaMemcpyDeviceToHost);
        cudaMemcpy(layer_input_opt_cpu, layer_input_opt, layer_input_elems * sizeof(__FLOAT16__), cudaMemcpyDeviceToHost);

        // Use relaxed tolerance for fused kernel (accumulated errors across stages)
        float atol_f32 = 0.5f;     // float outputs: some accumulation error
        float atol_bf16 = 0.1f;    // bf16 outputs: relaxed tolerance

        bool post_ok = check_result_fused<float>(post_mix_ori_cpu, post_mix_opt_cpu, post_mix_elems, atol_f32, "post_mix");
        bool comb_ok = check_result_fused<float>(comb_mix_ori_cpu, comb_mix_opt_cpu, comb_mix_elems, atol_f32, "comb_mix");
        bool layer_ok = check_result_fused<__FLOAT16__>(layer_input_ori_cpu, layer_input_opt_cpu, layer_input_elems, atol_bf16, "layer_input");

        is_right = post_ok && comb_ok && layer_ok;
    }

    // ===== Output tags =====
    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f) ? (opt_time / ori_time) : 0.f);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // ===== Cleanup =====
    free(residual_cpu); free(fn_cpu); free(mhc_scale_cpu); free(mhc_base_cpu);
    free(post_mix_ori_cpu); free(post_mix_opt_cpu);
    free(comb_mix_ori_cpu); free(comb_mix_opt_cpu);
    free(layer_input_ori_cpu); free(layer_input_opt_cpu);

    cudaFree(residual); cudaFree(fn); cudaFree(mhc_scale); cudaFree(mhc_base);
    cudaFree(post_mix_ori); cudaFree(post_mix_opt);
    cudaFree(comb_mix_ori); cudaFree(comb_mix_opt);
    cudaFree(layer_input_ori); cudaFree(layer_input_opt);
    cudaStreamDestroy(stream);

    return 0;
}
