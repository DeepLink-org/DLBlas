#pragma once
#include "helpers.cuh"

__global__ void sparse_attn_kernel_ori(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int* __restrict__ topk_idxs, const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out, int B, int M, int H, int D, int N, int K, float scale) {
    int idx = blockIdx.x;
    int total = B * M * H;
    if (idx >= total) return;
    int b = idx / (M * H);
    int rem = idx % (M * H);
    int m_idx = rem / H;
    int h_idx = rem % H;

    const uint16_t* q_bmh = q + ((b * M + m_idx) * H + h_idx) * D;
    uint16_t* out_bmh = out + ((b * M + m_idx) * H + h_idx) * D;
    const int* topk_bm = topk_idxs + (b * M + m_idx) * K;
    float sink = attn_sink[h_idx];
    int tid = threadIdx.x;

    // Step 1: Compute scores for K positions (K=16, fits in registers)
    float scores[16];
    float max_score = -1e30f;
    for (int k = 0; k < K; k++) {
        int kv_idx = topk_bm[k];
        if (kv_idx < 0) { scores[k] = -1e30f; continue; }
        const uint16_t* kv_n = kv + (b * N + kv_idx) * D;
        float dot = 0.f;
        for (int d = tid; d < D; d += blockDim.x) {
            dot += bf(q_bmh[d]) * bf(kv_n[d]);
        }
        // Reduce dot across threads
        extern __shared__ float smem[];
        smem[tid] = dot;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) smem[tid] += smem[tid+s]; __syncthreads(); }
        scores[k] = smem[0] * scale;
        max_score = fmaxf(max_score, scores[k]);
    }

    max_score = fmaxf(max_score, sink);

    // Step 2: Softmax
    float exp_sum = expf(sink - max_score);
    float attn_w[16];
    for (int k = 0; k < K; k++) {
        if (topk_bm[k] >= 0) {
            attn_w[k] = expf(scores[k] - max_score);
            exp_sum += attn_w[k];
        } else {
            attn_w[k] = 0.f;
        }
    }
    float inv_sum = 1.f / exp_sum;
    for (int k = 0; k < K; k++) attn_w[k] *= inv_sum;

    // Step 3: Weighted sum of KV
    for (int k = 0; k < K; k++) {
        if (topk_bm[k] < 0 || attn_w[k] == 0.f) continue;
        int kv_idx = topk_bm[k];
        const uint16_t* kv_n = kv + (b * N + kv_idx) * D;
        // Skip if we can't contribute (only writes - but we need read-modify-write)
    }

    // Initialize output to 0 then accumulate
    for (int d = tid; d < D; d += blockDim.x) {
        float oval = 0.f;
        for (int k = 0; k < K; k++) {
            if (topk_bm[k] >= 0) {
                int kv_idx = topk_bm[k];
                const uint16_t* kv_n = kv + (b * N + kv_idx) * D;
                oval += attn_w[k] * bf(kv_n[d]);
            }
        }
        out_bmh[d] = fb(oval);
    }
}
static void test_tmp_kernel_ori(const uint16_t* q,const uint16_t* kv,const int* topk,const float* sink,
    uint16_t* out,int B,int M,int H,int D,int N,int K,float scale,cudaStream_t s){
    int total=B*M*H, bs=64;
    sparse_attn_kernel_ori<<<total,bs,bs*sizeof(float),s>>>(q,kv,topk,sink,out,B,M,H,D,N,K,scale);
}