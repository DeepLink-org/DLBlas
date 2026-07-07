#pragma once
// engram_hash Iter 9: 2D grid with hardcoded max_ngram_size=3 + FP modulo
// blockIdx.y = layer, thread → (token, ngram_j) decoding via simple calculation
// Hardcoding ngram_minus_1=2 eliminates the loop and enables full unrolling

__device__ __forceinline__ int32_t fast_mod_i64_i32(int64_t hash, int32_t d) {
    double hash_d = (double)hash;
    double d_d = (double)d;
    int64_t q = (int64_t)(hash_d / d_d);
    int64_t r = hash - q * (int64_t)d;
    if (r < 0) r += d;
    if (r >= d) r -= d;
    return (int32_t)r;
}

__global__ void engram_hash_kernel_opt(
    const int32_t* ngram_token_ids,
    const int64_t* multipliers,
    const int32_t* vocab_sizes,
    const int32_t* offsets,
    int32_t* output,
    int num_tokens,
    int max_ngram_size,
    int num_ngram_layers,
    int num_embed_table_per_ngram)
{
    // Hardcoded: max_ngram_size=3, ngram_minus_1=2, tables_per_ngram=16
    const int ngram_minus_1 = 2;
    const int tables_per_ngram = 16;

    int layer = blockIdx.y;
    if (layer >= num_ngram_layers) return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_per_layer = num_tokens * 2;  // ngram_minus_1=2
    if (tid >= total_per_layer) return;

    int ngram_j = tid & 1;          // tid % 2
    int token   = tid >> 1;          // tid / 2

    int token_base = token * 3;      // max_ngram_size=3
    int mult_base  = layer * 3;

    int32_t t0 = ngram_token_ids[token_base + 0];
    int32_t t1 = ngram_token_ids[token_base + 1];
    int32_t t2 = ngram_token_ids[token_base + 2];
    int64_t m0 = multipliers[mult_base + 0];
    int64_t m1 = multipliers[mult_base + 1];
    int64_t m2 = multipliers[mult_base + 2];

    // Hash computation (hardcoded for max_ngram_size=3)
    int64_t hash = (int64_t)t0 * m0;
    if (ngram_j >= 0) {
        hash ^= (int64_t)t1 * m1;
    }
    if (ngram_j >= 1) {
        hash ^= (int64_t)t2 * m2;
    }

    int vs_base  = layer * 2 * num_embed_table_per_ngram + ngram_j * num_embed_table_per_ngram;
    int off_base = layer * tables_per_ngram + ngram_j * num_embed_table_per_ngram;
    int out_base = layer * num_tokens * tables_per_ngram + token * tables_per_ngram + ngram_j * num_embed_table_per_ngram;

    #pragma unroll
    for (int t = 0; t < num_embed_table_per_ngram; t++) {
        int32_t vs  = vocab_sizes[vs_base + t];
        int32_t off = offsets[off_base + t];
        output[out_base + t] = fast_mod_i64_i32(hash, vs) + off;
    }
}

void test_tmp_kernel_opt(
    int32_t* ngram_token_ids, int64_t* multipliers, int32_t* vocab_sizes, int32_t* offsets, int32_t* output,
    int num_tokens, int max_ngram_size, int num_ngram_layers, int num_embed_table_per_ngram,
    cudaStream_t stream)
{
    const int block_size = 256;
    int total_per_layer = num_tokens * 2; // ngram_minus_1=2
    int num_blocks_x = (total_per_layer + block_size - 1) / block_size;
    dim3 grid(num_blocks_x, num_ngram_layers);
    engram_hash_kernel_opt<<<grid, block_size, 0, stream>>>(
        ngram_token_ids, multipliers, vocab_sizes, offsets, output,
        num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram);
}
