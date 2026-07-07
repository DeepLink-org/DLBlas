#pragma once
// engram_hash check kernel (identical to baseline for correctness verification)
// Computes n-gram hash embedding indices via XOR-product-mod
__global__ void engram_hash_kernel_ori(
    const int32_t* __restrict__ ngram_token_ids,
    const int64_t* __restrict__ multipliers,
    const int32_t* __restrict__ vocab_sizes,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ output,
    int num_tokens,
    int max_ngram_size,
    int num_ngram_layers,
    int num_embed_table_per_ngram)
{
    int ngram_minus_1 = max_ngram_size - 1;
    int tables_per_ngram = ngram_minus_1 * num_embed_table_per_ngram;
    int total_elems = num_ngram_layers * num_tokens * tables_per_ngram;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    int table   = idx % num_embed_table_per_ngram;
    int tmp1    = idx / num_embed_table_per_ngram;
    int ngram_j = tmp1 % ngram_minus_1;
    int tmp2    = tmp1 / ngram_minus_1;
    int token   = tmp2 % num_tokens;
    int layer   = tmp2 / num_tokens;

    int64_t hash = (int64_t)ngram_token_ids[token * max_ngram_size + 0] * multipliers[layer * max_ngram_size + 0];
    for (int k = 1; k <= ngram_j + 1; k++) {
        hash ^= (int64_t)ngram_token_ids[token * max_ngram_size + k] * multipliers[layer * max_ngram_size + k];
    }

    int32_t vocab_size = vocab_sizes[layer * ngram_minus_1 * num_embed_table_per_ngram + ngram_j * num_embed_table_per_ngram + table];
    int32_t offset     = offsets[layer * tables_per_ngram + ngram_j * num_embed_table_per_ngram + table];

    output[idx] = (int32_t)(hash % (int64_t)vocab_size) + offset;
}

void test_tmp_kernel_ori(
    int32_t* ngram_token_ids, int64_t* multipliers, int32_t* vocab_sizes, int32_t* offsets, int32_t* output,
    int num_tokens, int max_ngram_size, int num_ngram_layers, int num_embed_table_per_ngram,
    cudaStream_t stream)
{
    int ngram_minus_1 = max_ngram_size - 1;
    int tables_per_ngram = ngram_minus_1 * num_embed_table_per_ngram;
    int total_elems = num_ngram_layers * num_tokens * tables_per_ngram;
    const int block_size = 256;
    int num_blocks = (total_elems + block_size - 1) / block_size;
    engram_hash_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        ngram_token_ids, multipliers, vocab_sizes, offsets, output,
        num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram);
}
