#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"

// engram_hash test harness
// Operator: n-gram hash embedding index computation
// Inputs:  ngram_token_ids [num_tokens, max_ngram_size] int32
//          multipliers [num_ngram_layers, max_ngram_size] int64
//          vocab_sizes [num_ngram_layers, max_ngram_size-1, num_embed_table_per_ngram] int32
//          offsets [num_ngram_layers, (max_ngram_size-1)*num_embed_table_per_ngram] int32
// Output:  embedding indices [num_ngram_layers, num_tokens, (max_ngram_size-1)*num_embed_table_per_ngram] int32

__global__ void warm_up() {}

// Hash-based random number generator for reproducibility
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int main(int argc, char *argv[]) {
    int warm_up_count = (argc > 1) ? atoi(argv[1]) : 5;
    int test_count    = (argc > 2) ? atoi(argv[2]) : 1000;
    int exec_mode     = (argc > 3) ? atoi(argv[3]) : 0;

    printf("<warm_up_count>%d</warm_up_count>\n", warm_up_count);
    printf("<test_count>%d</test_count>\n", test_count);
    printf("<exec_mode>%d</exec_mode>\n", exec_mode);

    CUDA_INIT();

    // Operator dimensions (from engram_hash.py test_engram_hash)
    const int num_tokens = 4096;
    const int max_ngram_size = 3;
    const int num_ngram_layers = 2;
    const int num_embed_table_per_ngram = 8;
    const int ngram_minus_1 = max_ngram_size - 1; // 2
    const int tables_per_ngram = ngram_minus_1 * num_embed_table_per_ngram; // 16
    const int output_elems = num_ngram_layers * num_tokens * tables_per_ngram; // 2 * 4096 * 16 = 131072

    const int tokens_size    = num_tokens * max_ngram_size;          // 4096 * 3 = 12288
    const int mult_size      = num_ngram_layers * max_ngram_size;    // 2 * 3 = 6
    const int vocab_size_elems = num_ngram_layers * ngram_minus_1 * num_embed_table_per_ngram; // 2 * 2 * 8 = 32
    const int offset_size    = num_ngram_layers * tables_per_ngram;  // 2 * 16 = 32
    const int output_size    = output_elems;                         // 131072

    printf("<num_tokens>%d</num_tokens>\n", num_tokens);
    printf("<max_ngram_size>%d</max_ngram_size>\n", max_ngram_size);
    printf("<num_ngram_layers>%d</num_ngram_layers>\n", num_ngram_layers);
    printf("<num_embed_table_per_ngram>%d</num_embed_table_per_ngram>\n", num_embed_table_per_ngram);
    printf("<total_output_elems>%d</total_output_elems>\n", output_elems);

    // ===== Allocate host memory =====
    int32_t* ngram_token_ids_cpu = (int32_t*)malloc(tokens_size * sizeof(int32_t));
    int64_t* multipliers_cpu     = (int64_t*)malloc(mult_size * sizeof(int64_t));
    int32_t* vocab_sizes_cpu     = (int32_t*)malloc(vocab_size_elems * sizeof(int32_t));
    int32_t* offsets_cpu         = (int32_t*)malloc(offset_size * sizeof(int32_t));
    int32_t* Y_ori_cpu           = (int32_t*)malloc(output_size * sizeof(int32_t));
    int32_t* Y_opt_cpu           = (int32_t*)malloc(output_size * sizeof(int32_t));

    // ===== Initialize inputs with deterministic random values =====
    uint32_t rng_state = 42;
    for (int i = 0; i < tokens_size; i++) {
        ngram_token_ids_cpu[i] = (int32_t)(xorshift32(&rng_state) % 100000);
    }
    for (int i = 0; i < mult_size; i++) {
        multipliers_cpu[i] = (int64_t)(xorshift32(&rng_state) % 100000);
    }
    for (int i = 0; i < vocab_size_elems; i++) {
        vocab_sizes_cpu[i] = (int32_t)(100000 + xorshift32(&rng_state) % 900000); // [100000, 1000000)
    }
    // Compute offsets as exclusive prefix sum of vocab_sizes
    for (int l = 0; l < num_ngram_layers; l++) {
        int32_t cumsum = 0;
        for (int j = 0; j < ngram_minus_1 * num_embed_table_per_ngram; j++) {
            int off_idx = l * tables_per_ngram + j;
            int vs_idx  = l * ngram_minus_1 * num_embed_table_per_ngram + j;
            offsets_cpu[off_idx] = cumsum;
            cumsum += vocab_sizes_cpu[vs_idx];
        }
    }

    // ===== Allocate device memory =====
    int32_t *ngram_token_ids_dev, *vocab_sizes_dev, *offsets_dev, *Y_ori, *Y_opt;
    int64_t *multipliers_dev;

    cudaMalloc((void**)&ngram_token_ids_dev, tokens_size * sizeof(int32_t));
    cudaMalloc((void**)&multipliers_dev,     mult_size * sizeof(int64_t));
    cudaMalloc((void**)&vocab_sizes_dev,     vocab_size_elems * sizeof(int32_t));
    cudaMalloc((void**)&offsets_dev,         offset_size * sizeof(int32_t));
    cudaMalloc((void**)&Y_ori,              output_size * sizeof(int32_t));
    cudaMalloc((void**)&Y_opt,              output_size * sizeof(int32_t));

    // Copy inputs to device
    cudaMemcpy(ngram_token_ids_dev, ngram_token_ids_cpu, tokens_size * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(multipliers_dev, multipliers_cpu, mult_size * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(vocab_sizes_dev, vocab_sizes_cpu, vocab_size_elems * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(offsets_dev, offsets_cpu, offset_size * sizeof(int32_t), cudaMemcpyHostToDevice);

    float ori_time = 0.f, opt_time = 0.f;

    // ===== Run ori kernel (exec_mode 0 or 1) =====
    if (exec_mode == 0 || exec_mode == 1) {
        for (int i = 0; i < warm_up_count; i++)
            test_tmp_kernel_ori(ngram_token_ids_dev, multipliers_dev, vocab_sizes_dev, offsets_dev, Y_ori,
                               num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_ori(ngram_token_ids_dev, multipliers_dev, vocab_sizes_dev, offsets_dev, Y_ori,
                               num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram, stream);
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
            test_tmp_kernel_opt(ngram_token_ids_dev, multipliers_dev, vocab_sizes_dev, offsets_dev, Y_opt,
                               num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram, stream);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float total_time = 0.f;
        for (int i = 0; i < test_count; i++) {
            cudaEventRecord(start, stream);
            test_tmp_kernel_opt(ngram_token_ids_dev, multipliers_dev, vocab_sizes_dev, offsets_dev, Y_opt,
                               num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram, stream);
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
        cudaMemcpy(Y_ori_cpu, Y_ori, output_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(Y_opt_cpu, Y_opt, output_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Exact int32 comparison
        int diff_nums = 0;
        for (int i = 0; i < output_size; i++) {
            if (Y_ori_cpu[i] != Y_opt_cpu[i]) {
                diff_nums++;
                if (diff_nums <= 10)
                    printf("Mismatch at %d: ref=%d, test=%d\n", i, Y_ori_cpu[i], Y_opt_cpu[i]);
            }
        }
        if (diff_nums > 0) {
            printf("result is not right, %d mismatches\n", diff_nums);
            is_right = false;
        } else {
            printf("result is right\n");
        }
    }

    // ===== Output tags =====
    printf("<time_before_opt>%f ms</time_before_opt>\n", ori_time);
    printf("<time_after_opt>%f ms</time_after_opt>\n", opt_time);
    printf("<runtime_ratio> %f</runtime_ratio>\n", (ori_time > 0.f) ? (opt_time / ori_time) : 0.f);
    printf("<precision>%s</precision>\n", is_right ? "True" : "False");

    // ===== Cleanup =====
    free(ngram_token_ids_cpu); free(multipliers_cpu); free(vocab_sizes_cpu); free(offsets_cpu);
    free(Y_ori_cpu); free(Y_opt_cpu);

    cudaFree(ngram_token_ids_dev); cudaFree(multipliers_dev); cudaFree(vocab_sizes_dev); cudaFree(offsets_dev);
    cudaFree(Y_ori); cudaFree(Y_opt);
    cudaStreamDestroy(stream);

    return 0;
}
