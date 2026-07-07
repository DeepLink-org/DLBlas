// ============================================================================
// indexer baseline kernel (ori)
// ============================================================================
// Semantics:
//   index_score = einsum("bshd,btd->bsht", q, kv_cache)   // dot product per (b,s,h,t)
//   index_score = relu(index_score) * weights.unsqueeze(-1) // activate & scale
//   index_score = index_score.sum(dim=2)                   // reduce over heads → [B,S,T]
//   Apply causal mask (if start_pos==0)
//   topk_idxs = topk(index_score, TopK)                    // select top K indices
//
// q:        [B, S, H, D]       bfloat16
// kv_cache: [B, T_total, D]    bfloat16
// weights:  [B, S, H]          bfloat16
// topk_idxs:[B, S, TopK]       int32 output
// ============================================================================

__global__ void indexer_kernel_ori(
    const __BFLOAT16__* __restrict__ q,
    const __BFLOAT16__* __restrict__ kv_cache,
    const __BFLOAT16__* __restrict__ weights,
    int* __restrict__ topk_idxs,
    int B, int S, int H, int D, int T_total, int T_used, int TopK, int start_pos)
{
    int bs = blockIdx.x;           // 0 .. B*S - 1
    int b  = bs / S;
    int s  = bs % S;
    int tid = threadIdx.x;

    if (b >= B) return;

    extern __shared__ float s_scores[];  // [T_used] float scores

    // ---- Initialize scores to 0 ------------------------------------------
    for (int t = tid; t < T_used; t += blockDim.x) {
        s_scores[t] = 0.0f;
    }
    __syncthreads();

    // ---- Compute scores: thread-parallel over (h, t) pairs --------------
    int total_ht = H * T_used;
    for (int idx = tid; idx < total_ht; idx += blockDim.x) {
        int h = idx / T_used;
        int t = idx % T_used;

        // Compute dot product: q[b,s,h,:] · kv_cache[b,t,:]
        float dot = 0.0f;
        int q_offs  = ((b * S + s) * H + h) * D;
        int kv_offs = (b * T_total + t) * D;

        for (int d = 0; d < D; d++) {
            dot += __bfloat162float(q[q_offs + d]) *
                   __bfloat162float(kv_cache[kv_offs + d]);
        }

        // ReLU activation
        if (dot > 0.0f) {
            float w = __bfloat162float(weights[((b * S + s) * H + h)]);
            // Accumulate: using float for precision
            atomicAdd(&s_scores[t], dot * w);
        }
    }
    __syncthreads();

    // ---- Causal mask (only for start_pos == 0) ---------------------------
    // Mask logic: position t is causal-masked if t >= ceil((s+1)/ratio)
    // where ratio = 4. Simplified: t >= (s + 1 + 3) / 4 = (s + 4) / 4
    if (start_pos == 0) {
        int ratio = 4;  // compress_ratio
        int visible_limit = (s + 1 + ratio - 1) / ratio;  // ceil((s+1)/ratio)
        for (int t = tid; t < T_used; t += blockDim.x) {
            if (t >= visible_limit) {
                s_scores[t] = -1e30f;
            }
        }
    }
    __syncthreads();

    // ---- TopK selection --------------------------------------------------
    // Simple approach: K passes to find top K indices
    // For thread 0 only (or few threads cooperating)
    // Mark selected positions with -INF after each selection

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
                s_scores[best_t] = -1e30f;  // Mark as used
            }
        }
        // Fill remaining slots with -1
        for (int k = actual_k; k < TopK; k++) {
            topk_idxs[(bs * TopK) + k] = -1;
        }
    }
}


// ============================================================================
// Host wrapper
// ============================================================================
template <typename T>
void test_tmp_kernel_ori(
    T* q, T* kv_cache, T* weights, int* topk_idxs,
    int B, int S, int H, int D, int T_total, int T_used, int TopK,
    int start_pos,
    cudaStream_t stream)
{
    int total_blocks = B * S;

    // Choose block size: next power of two >= max(T_used, D), capped at 256
    int block_size = (T_used > D) ? T_used : D;
    if (block_size > 256) block_size = 256;
    if (block_size < 64)  block_size = 64;
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    if (bs > 256) bs = 256;
    block_size = bs;

    int smem = T_used * sizeof(float);

    indexer_kernel_ori<<<total_blocks, block_size, smem, stream>>>(
        (const __BFLOAT16__*)q, (const __BFLOAT16__*)kv_cache,
        (const __BFLOAT16__*)weights, topk_idxs,
        B, S, H, D, T_total, T_used, TopK, start_pos);
}
