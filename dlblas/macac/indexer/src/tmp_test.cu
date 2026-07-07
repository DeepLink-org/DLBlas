#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"


// ============================================================================
// Custom precision checker for int32 indices
// ============================================================================
template<typename T>
bool check_result_idx(T* ref, T* test, int num_elements) {
    int diff_nums = 0;
    for (int i = 0; i < num_elements; i++) {
        if (ref[i] != test[i]) {
            diff_nums++;
            if (diff_nums < 10)
                printf("[idx] Mismatch at %d: ref=%d, test=%d\n",
                       i, (int)ref[i], (int)test[i]);
        }
    }
    if (diff_nums > 0) {
        printf("[idx] result is not right, %d mismatches\n", diff_nums);
        return false;
    }
    printf("[idx] result is right\n");
    return true;
}


struct ResultStruct {
    float ori_time;
    float opt_time;
    bool  result_is_right;
};

__global__ void warm_up() {}


// ============================================================================
// Test driver for indexer — both ori and opt kernels share the same inputs.
// ============================================================================
template <typename T>
ResultStruct test_indexer(
    int B, int S, int H, int D, int T_total, int T_used, int TopK,
    int start_pos,
    int warm_up_count, int test_count, int exec_mode)
{
    ResultStruct result;
    result.ori_time = 0.0f;
    result.opt_time = 0.0f;
    result.result_is_right = true;

    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ---- Element counts ----
    int q_elems       = B * S * H * D;
    int kv_elems      = B * T_total * D;
    int weights_elems = B * S * H;
    int idx_elems     = B * S * TopK;

    // ---- Allocate GPU memory ----
    T   *q_d = nullptr, *kv_d = nullptr, *weights_d = nullptr;
    int *idx_ori_d = nullptr, *idx_opt_d = nullptr;

    cudaMalloc((void**)&q_d,        sizeof(T)   * q_elems);
    cudaMalloc((void**)&kv_d,       sizeof(T)   * kv_elems);
    cudaMalloc((void**)&weights_d,  sizeof(T)   * weights_elems);
    cudaMalloc((void**)&idx_ori_d,  sizeof(int) * idx_elems);
    cudaMalloc((void**)&idx_opt_d,  sizeof(int) * idx_elems);

    // ---- Allocate CPU memory ----
    T   *q_cpu      = (T*)malloc(sizeof(T)   * q_elems);
    T   *kv_cpu     = (T*)malloc(sizeof(T)   * kv_elems);
    T   *weights_cpu = (T*)malloc(sizeof(T)  * weights_elems);
    int *idx_ori_cpu = (int*)malloc(sizeof(int) * idx_elems);
    int *idx_opt_cpu = (int*)malloc(sizeof(int) * idx_elems);

    // ---- Initialize inputs ----
    // q: pseudo-random-like bf16 (i*7 mod 127)
    for (int i = 0; i < q_elems; i++) {
        q_cpu[i] = (T)((float)((i * 7) % 127) / 127.0f);
    }
    // kv_cache: different pattern for diversity
    for (int i = 0; i < kv_elems; i++) {
        kv_cpu[i] = (T)((float)((i * 11) % 131) / 131.0f);
    }
    // weights: scale-like values around 1.0
    for (int i = 0; i < weights_elems; i++) {
        weights_cpu[i] = (T)(0.5f + ((float)((i * 3) % 100)) / 100.0f);
    }

    // ---- Copy to GPU ----
    cudaMemcpy(q_d,        q_cpu,        sizeof(T)   * q_elems,        cudaMemcpyHostToDevice);
    cudaMemcpy(kv_d,       kv_cpu,       sizeof(T)   * kv_elems,       cudaMemcpyHostToDevice);
    cudaMemcpy(weights_d,  weights_cpu,  sizeof(T)   * weights_elems,  cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    // ========================================================================
    // exec_mode 0,1: run ori kernel
    // ========================================================================
    if (exec_mode == 0 || exec_mode == 1) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_ori<T>(q_d, kv_d, weights_d, idx_ori_d,
                                    B, S, H, D, T_total, T_used, TopK,
                                    start_pos, stream);
        }
        cudaDeviceSynchronize();

        float total_time = 0.0f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori<T>(q_d, kv_d, weights_d, idx_ori_d,
                                    B, S, H, D, T_total, T_used, TopK,
                                    start_pos, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);

            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;
        }
        result.ori_time = total_time / (float)test_count;
        printf("origin fprop average time: %f ms\n", result.ori_time);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ========================================================================
    // exec_mode 0,2: run opt kernel
    // ========================================================================
    if (exec_mode == 0 || exec_mode == 2) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int i = 0; i < warm_up_count; i++) {
            test_tmp_kernel_opt<T>(q_d, kv_d, weights_d, idx_opt_d,
                                    B, S, H, D, T_total, T_used, TopK,
                                    start_pos, stream);
        }
        cudaDeviceSynchronize();

        float total_time = 0.0f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt<T>(q_d, kv_d, weights_d, idx_opt_d,
                                    B, S, H, D, T_total, T_used, TopK,
                                    start_pos, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);

            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;
        }
        result.opt_time = total_time / (float)test_count;
        printf("opt   fprop average time: %f ms\n", result.opt_time);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaDeviceSynchronize();

    // ========================================================================
    // Precision check (only exec_mode 0)
    // ========================================================================
    if (exec_mode == 0) {
        cudaMemcpy(idx_ori_cpu, idx_ori_d, sizeof(int) * idx_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(idx_opt_cpu, idx_opt_d, sizeof(int) * idx_elems, cudaMemcpyDeviceToHost);
        result.result_is_right = check_result_idx<int>(idx_ori_cpu, idx_opt_cpu, idx_elems);
    } else {
        result.result_is_right = true;
    }

    // ---- Cleanup ----
    free(q_cpu);        free(kv_cpu);
    free(weights_cpu);  free(idx_ori_cpu);
    free(idx_opt_cpu);

    cudaFree(q_d);       cudaFree(kv_d);
    cudaFree(weights_d); cudaFree(idx_ori_d);
    cudaFree(idx_opt_d);

    cudaStreamDestroy(stream);

    return result;
}


// ============================================================================
// main
// ============================================================================
int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    // Default shape (matches origin-copy/indexer.py get_inputs()):
    //   args: B=2, S=64, H=16 (index_n_heads), D=64 (index_head_dim)
    //   max_seq_len=1024, compress_ratio=4 → T_total=256
    //   start_pos=0, seqlen=64 → T_used=64/4=16
    //   TopK=min(128, T_used)=16
    int B        = 2;
    int S        = 64;
    int H        = 16;
    int D        = 64;
    int T_total  = 256;
    int T_used   = 16;
    int TopK     = 16;
    int start_pos = 0;

    printf("Shape: B=%d S=%d H=%d D=%d T_total=%d T_used=%d TopK=%d start_pos=%d\n",
           B, S, H, D, T_total, T_used, TopK, start_pos);

    // Use bfloat16 type
    ResultStruct result = test_indexer<__BFLOAT16__>(
        B, S, H, D, T_total, T_used, TopK, start_pos,
        warm_up_count, test_count, exec_mode);

    float ratio = (result.ori_time > 0.0f) ? result.opt_time / result.ori_time : 0.0f;

    printf("<time_before_opt>%f ms</time_before_opt>\n", result.ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n",  result.opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", ratio);
    printf("<precision>%s</precision>\n", result.result_is_right ? "True" : "False");

    return 0;
}
