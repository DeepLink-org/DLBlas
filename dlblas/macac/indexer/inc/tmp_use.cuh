// ============================================================================
// indexer optimized kernel (opt) — FINAL: kv preload + 4x unroll + block 256
// ============================================================================
// Best version from Iteration 4
// Key optimizations:
//   1. Preload kv_cache into shared memory → avoids H=16 redundant loads per row
//   2. 4x unrolled dot product
//   3. Block size 256 for occupancy
// ============================================================================

__global__ void indexer_kernel_opt(
    const __BFLOAT16__* __restrict__ q,
    const __BFLOAT16__* __restrict__ kv_cache,
    const __BFLOAT16__* __restrict__ weights,
    int* __restrict__ topk_idxs,
    int B, int S, int H, int D, int T_total, int T_used, int TopK, int start_pos)
{
    int bs = blockIdx.x;
    int b  = bs / S;
    int s  = bs % S;
    int tid = threadIdx.x;

    if (b >= B) return;

    // Shared memory:
    //   s_kv[T_used * D]     — preloaded kv_cache as unsigned short (bf16)
    //   s_scores[T_used]     — final scores as float
    extern __shared__ float s_buf[];
    unsigned short* s_kv = (unsigned short*)s_buf;
    float* s_scores       = (float*)(s_kv + T_used * D);

    // ---- Preload kv_cache[b, 0:T_used, :] into shared memory -------------
    int kv_b_base = b * T_total * D;
    int kv_elems  = T_used * D;
    for (int i = tid; i < kv_elems; i += blockDim.x) {
        s_kv[i] = kv_cache[kv_b_base + i];
    }
    __syncthreads();

    // ---- Initialize scores -----------------------------------------------
    for (int t = tid; t < T_used; t += blockDim.x) {
        s_scores[t] = 0.0f;
    }
    __syncthreads();

    // ---- Precompute offsets ----------------------------------------------
    int bs_H_D = ((b * S + s) * H) * D;
    int bs_H   = (b * S + s) * H;

    // ---- Compute scores: thread-parallel over (h, t) pairs ---------------
    int total_ht = H * T_used;
    for (int idx = tid; idx < total_ht; idx += blockDim.x) {
        int h = idx / T_used;
        int t = idx % T_used;

        float dot = 0.0f;
        int q_offs = bs_H_D + h * D;
        int kv_offs = t * D;  // shared memory offset

        #pragma unroll
        for (int d = 0; d < D; d += 4) {
            dot += __bfloat162float(q[q_offs + d]) *
                   __bfloat162float(s_kv[kv_offs + d]);
            dot += __bfloat162float(q[q_offs + d + 1]) *
                   __bfloat162float(s_kv[kv_offs + d + 1]);
            dot += __bfloat162float(q[q_offs + d + 2]) *
                   __bfloat162float(s_kv[kv_offs + d + 2]);
            dot += __bfloat162float(q[q_offs + d + 3]) *
                   __bfloat162float(s_kv[kv_offs + d + 3]);
        }

        if (dot > 0.0f) {
            float w = __bfloat162float(weights[bs_H + h]);
            atomicAdd(&s_scores[t], dot * w);
        }
    }
    __syncthreads();

    // ---- Causal mask -----------------------------------------------------
    if (start_pos == 0) {
        int ratio = 4;
        int visible_limit = (s + 1 + ratio - 1) / ratio;
        for (int t = tid; t < T_used; t += blockDim.x) {
            if (t >= visible_limit) {
                s_scores[t] = -1e30f;
            }
        }
    }
    __syncthreads();

    // ---- TopK selection --------------------------------------------------
    if (tid == 0) {
        int actual_k = (TopK < T_used) ? TopK : T_used;
        for (int k = 0; k < actual_k; k++) {
            float best_score = -1e30f;
            int   best_t     = -1;
            for (int t = 0; t < T_used; t++) {
                if (s_scores[t] > best_score) {
                    best_score = s_scores[t];
                    best_t     = t;
                }
            }
            topk_idxs[(bs * TopK) + k] = best_t;
            if (best_t >= 0) {
                s_scores[best_t] = -1e30f;
            }
        }
        for (int k = actual_k; k < TopK; k++) {
            topk_idxs[(bs * TopK) + k] = -1;
        }
    }
}


template <typename T>
void test_tmp_kernel_opt(
    T* q, T* kv_cache, T* weights, int* topk_idxs,
    int B, int S, int H, int D, int T_total, int T_used, int TopK,
    int start_pos,
    cudaStream_t stream)
{
    int total_blocks = B * S;
    int block_size = 256;

    // Shared memory: T_used*D unsigned shorts (kv) + T_used floats (scores)
    int smem = T_used * D * sizeof(unsigned short) + T_used * sizeof(float);

    indexer_kernel_opt<<<total_blocks, block_size, smem, stream>>>(
        (const __BFLOAT16__*)q, (const __BFLOAT16__*)kv_cache,
        (const __BFLOAT16__*)weights, topk_idxs,
        B, S, H, D, T_total, T_used, TopK, start_pos);
}
