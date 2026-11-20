// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
#ifndef TECO_UAL_KERNEL_FLASH_ATTENTION_FLASH_ATTENTION_COMMON_H_
#define TECO_UAL_KERNEL_FLASH_ATTENTION_FLASH_ATTENTION_COMMON_H_

#include <simd.h>
#include <slave.h>
#include <sdaa_transpose.h>

#include "ual/com/rt.h"
#include "ual/kernel/flash_attention/flash_attention.h"
#include "teco_hal.h"  // NOLINT(build/include_subdir)

using namespace sdaa;  //NOLINT
using namespace teco::hal;  //NOLINT

// macro definition of MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
// macro definition of MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// FLT_MAX if not already defined
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

// load a tile from KV cache (key or value) into dst
// src = cache[block_id][kv_head_idx][0..num_tokens-1][0..head_size-1]
// dst = [num_tokens, head_size]
static __device__ __always_inline void load_kv_cache(
    half *dst, const half *cache, const int *block_table, int logical_block,
    int kv_head_idx, int kv_head_num, int block_size, int head_size, int num_tokens) {
    int block_id = block_table[logical_block];
    const half *src = cache
        + block_id * kv_head_num * block_size * head_size
        + kv_head_idx * block_size * head_size;
    memcpy(dst, src, num_tokens * head_size * sizeof(half));
}

static __device__ __always_inline void fill_inf(float *buf, int len) {
    // if use simd should assert aligned
    int neg_inf_bit = 0xFF800000;
    float neg_inf;
    memcpy(&neg_inf, &neg_inf_bit, sizeof(float));
    for (int i = 0; i < len; i++) {
        buf[i] = neg_inf;
    }
}

// causal mask on score tile [cur_BM, cur_BN] inside [BM, BN]
static __device__ __always_inline void apply_causal_mask(
    float *score, int cur_BM, int cur_BN,
    int qi_off, int kj_off, int q_seq_len, int k_seq_len,
    int BN) {

    int q_start = k_seq_len - q_seq_len;
    if (q_start < 0) q_start = 0;

    for (int qi = 0; qi < cur_BM; qi++) {
        int rel_q_pos = q_start + qi_off + qi;
        int allowed = rel_q_pos - kj_off + 1;
        if (allowed < 0) allowed = 0;
        if (allowed >= cur_BN) continue;
        fill_inf(score + qi * BN + allowed, cur_BN - allowed);
    }
}

static __device__ __always_inline void apply_causal_mask(
    half *score, int cur_BM, int cur_BN,
    int qi_off, int kj_off, int q_seq_len, int k_seq_len,
    int BN) {
    int q_start = k_seq_len - q_seq_len;
    if (q_start < 0) q_start = 0;

    half neg_inf = (half)-FLT_MAX;
    for (int qi = 0; qi < cur_BM; qi++) {
        int rel_q_pos = q_start + qi_off + qi;
        int allowed = rel_q_pos - kj_off + 1;
        if (allowed < 0) allowed = 0;
        if (allowed >= cur_BN) continue;
        half *row = score + qi * BN;
        for (int col = allowed; col < cur_BN; col++) {
            row[col] = neg_inf;
        }
    }
}

static __device__ float compute_max(float *input, int size) {
    floatv16 vf = -FLT_MAX;
    floatv16 vres = -FLT_MAX;
    float result = -FLT_MAX;
    int i = 0;

    for (i = 0; i + 16 <= size; i += 16) {
        simd_load(vf, input + i);
        vres = simd_max(vf, vres);
    }

    for (; i < size; i++) {
        result = MAX(result, input[i]);
    }

    float *tmp = (float *)&vres;  // NOLINT(readability/casting)
    for (int j = 0; j < 16; j++) {
        result = MAX(result, tmp[j]);
    }
    return result;
}

static __device__ void online_softmax_step(
    float *scores,           // S: [BM, BN] scores -> exp(S - m_new)
    const float *pre_max,    // m_old: [m]
    const float *pre_sum,    // l_old: [m]
    float *cur_max,          // m_new: [m]
    float *cur_sum,          // l_new: [m]
    float *scale,            // scale = exp(m_old - m_new): [m]
    int m, int n,            // cur_BM, cur_BN
    int ld_scores) {         // BN

    // Step 1: per-row : cal m_new, scale
    for (int i = 0; i < m; i++) {
        float *src_row = scores + i * ld_scores;
        float m_block = compute_max(src_row, n);
        float m_new = MAX(pre_max[i], m_block);
        m_new = MAX(m_new, 0.0f);
        cur_max[i] = m_new;
        scale[i] = pre_max[i] == m_new
            ? 1.0f : expf(pre_max[i] - m_new);
    }

    // Step 2: inplace scores: S -> exp(S - m_new), cal l_new
    for (int i = 0; i < m; i++) {
        float *src_row = scores + i * ld_scores;
        float m_new = cur_max[i];
        float l_old = pre_sum[i];
        float s = scale[i];

        float l_block = 0.0f;
        int j = 0;
        for (; j + 16 <= n; j += 16) {
            floatv16 v;
            simd_load(v, src_row + j);
            v = simd_exp(v - m_new);
            simd_store(v, src_row + j);
            float *vector_values = (float *)&v;  // NOLINT(readability/casting)
            for (int t = 0; t < 16; t++) {
                l_block += vector_values[t];
            }
        }
        for (; j < n; j++) {
            src_row[j] = expf(src_row[j] - m_new);
            l_block += src_row[j];
        }
        cur_sum[i] = l_old * s + l_block;
    }
}


constexpr int FLASH_ATTN_BM = 128;
constexpr int FLASH_ATTN_BN = 32;

struct FlashAttentionSpmBuffers {
    int *block_table;
    half *q;
    half *k_tmp;
    half *k;
    float *score;
    half *out;
    float *out_accum;
    float *l_prev;
    float *m_prev;
    float *l_cur;
    float *m_cur;
    float *scale;
    half *p_score;
};


static __device__ __always_inline void allocate_flash_attention_buffers(
    const FlashAttentionArgs &args, FlashAttentionSpmBuffers *buffers) {
    int token_size = args.size_per_head * sizeof(half);
    buffers->block_table =
        (int *)rt_spm_malloc(  // NOLINT(readability/casting)
            args.block_table_dim * sizeof(int));
    buffers->q = (half *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * token_size);
    buffers->k_tmp = (half *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BN * token_size);
    buffers->k = (half *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BN * token_size);
    buffers->score =
        (float *)rt_spm_malloc(  // NOLINT(readability/casting)
            FLASH_ATTN_BM * FLASH_ATTN_BN * sizeof(float));
    buffers->out = (half *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * token_size);
    buffers->out_accum =
        (float *)rt_spm_malloc(  // NOLINT(readability/casting)
            FLASH_ATTN_BM * args.size_per_head * sizeof(float));
    buffers->l_prev = (float *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * sizeof(float));
    buffers->m_prev = (float *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * sizeof(float));
    buffers->l_cur = (float *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * sizeof(float));
    buffers->m_cur = (float *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * sizeof(float));
    buffers->scale = (float *)rt_spm_malloc(  // NOLINT(readability/casting)
        FLASH_ATTN_BM * sizeof(float));
    buffers->p_score =
        (half *)rt_spm_malloc(  // NOLINT(readability/casting)
            FLASH_ATTN_BM * FLASH_ATTN_BN * sizeof(half));
}


static __device__ void compute_flash_attention_tile(
    const FlashAttentionArgs &args, FlashAttentionSpmBuffers *buffers,
    int q_base, int q_seq_len, int k_seq_len, int q_offset, int head_idx) {
    int BM = FLASH_ATTN_BM;
    int BN = FLASH_ATTN_BN;
    int BK = args.size_per_head;
    int token_size = args.size_per_head * sizeof(half);
    int num_kv_groups = args.local_head_num / args.local_kv_head_num;
    int cur_BM = MIN(BM, q_seq_len - q_offset);

    for (int idx = 0; idx < cur_BM * BK; idx++) {
        buffers->out_accum[idx] = 0.0f;
    }
    for (int row = 0; row < cur_BM; row++) {
        buffers->l_prev[row] = 0.0f;
    }
    fill_inf(buffers->m_prev, cur_BM);
    float *l_ptr = buffers->l_prev;
    float *l_next = buffers->l_cur;
    float *m_ptr = buffers->m_prev;
    float *m_next = buffers->m_cur;

    half *src_q_ptr = (half *)args.q +  // NOLINT(readability/casting)
                      (q_base + q_offset) * (args.local_head_num * args.size_per_head) +
                      head_idx * args.size_per_head;
    size_t q_src_stride = (args.local_head_num - 1) * token_size;
    memcpy_stride(buffers->q, src_q_ptr, cur_BM * token_size, token_size,
                  {0, q_src_stride});

    int kv_head_idx = head_idx / num_kv_groups;
    int q_start = MAX(k_seq_len - q_seq_len, 0);
    int k_limit = MIN(k_seq_len, q_start + q_offset + cur_BM);
    for (int j = 0; j < k_limit; j += BN) {
        int cur_BN = MIN(BN, k_limit - j);
        int cur_block = j / args.block_size;
        load_kv_cache(buffers->k_tmp, (const half *)args.key_cache,
                      buffers->block_table, cur_block, kv_head_idx,
                      args.local_kv_head_num, args.block_size,
                      args.size_per_head, cur_BN);

        for (int t = 0; t < cur_BN; t++) {
            for (int h = 0; h < args.size_per_head; h++) {
                buffers->k[h * BN + t] =
                    buffers->k_tmp[t * args.size_per_head + h];
            }
        }
        if (cur_BN < BN) {
            for (int h = 0; h < args.size_per_head; h++) {
                for (int c = cur_BN; c < BN; c++) {
                    buffers->k[h * BN + c] = 0;
                }
            }
        }

        {
            MatmulConfig matmul_cfg;
            Matmul matmul(matmul_cfg);
            auto tensorQ = make_tensor<2>(
                buffers->q,
                {static_cast<size_t>(cur_BM),
                 static_cast<size_t>(args.size_per_head)});
            auto tensorK = make_tensor<2>(
                buffers->k,
                {static_cast<size_t>(args.size_per_head),
                 static_cast<size_t>(BN)});
            auto tensorC = make_tensor<2>(
                buffers->score,
                {static_cast<size_t>(cur_BM), static_cast<size_t>(BN)});
            matmul.compute_async<MatmulBlockConfig::M128_N32>(
                tensorQ, tensorK, /*is_lastk=*/true);
            matmul.storeC_spm_async(tensorC);
            matmul.wait_storeC(1);
        }

        {
            floatv16 vscale = args.softmax_scale;
            int total = cur_BM * BN;
            int si = 0;
            for (; si + 16 <= total; si += 16) {
                floatv16 v;
                simd_load(v, buffers->score + si);
                v = simd_mul(v, vscale);
                simd_store(v, buffers->score + si);
            }
            for (; si < total; si++) {
                buffers->score[si] *= args.softmax_scale;
            }
        }

        apply_causal_mask(buffers->score, cur_BM, cur_BN, q_offset, j,
                          q_seq_len, k_seq_len, BN);
        online_softmax_step(buffers->score, m_ptr, l_ptr, m_next, l_next,
                            buffers->scale, cur_BM, cur_BN, BN);

        for (int ri = 0; ri < cur_BM; ri++) {
            floatv16 vscale = buffers->scale[ri];
            float *out_row = buffers->out_accum + ri * BK;
            int rk = 0;
            for (; rk + 16 <= BK; rk += 16) {
                floatv16 value;
                simd_load(value, out_row + rk);
                value = simd_mul(value, vscale);
                simd_store(value, out_row + rk);
            }
            for (; rk < BK; rk++) {
                out_row[rk] *= buffers->scale[ri];
            }
        }

        {
            int total = cur_BM * BN;
            int si = 0;
            for (; si + 16 <= total; si += 16) {
                floatv16 vf;
                halfv16 vh;
                simd_load(vf, buffers->score + si);
                vh = (halfv16)vf;
                simd_store(vh, buffers->p_score + si);
            }
            for (; si < total; si++) {
                buffers->p_score[si] = (half)buffers->score[si];
            }
        }
        for (int ri = 0; ri < cur_BM; ri++) {
            for (int cj = cur_BN; cj < BN; cj++) {
                buffers->p_score[ri * BN + cj] = 0;
            }
        }

        int block_id = buffers->block_table[cur_block];
        const half *v_cache_base = (const half *)args.value_cache
            + block_id * args.local_kv_head_num * args.block_size * args.size_per_head
            + kv_head_idx * args.block_size * args.size_per_head;
        int gemm_bn = 32;
        for (int hb = 0; hb < BK / gemm_bn; hb++) {
            memcpy_stride(buffers->k, v_cache_base + hb * gemm_bn,
                          cur_BN * gemm_bn * sizeof(half),
                          gemm_bn * sizeof(half),
                          {0, (args.size_per_head - gemm_bn) * sizeof(half)});
            if (cur_BN < BN) {
                for (int r = cur_BN * gemm_bn; r < BN * gemm_bn; r++) {
                    buffers->k[r] = 0;
                }
            }

            MatmulConfig matmul_cfg;
            Matmul matmul(matmul_cfg);
            auto tensorP = make_tensor<2>(
                buffers->p_score,
                {static_cast<size_t>(cur_BM), static_cast<size_t>(BN)});
            auto tensorV = make_tensor<2>(
                buffers->k,
                {static_cast<size_t>(BN), static_cast<size_t>(gemm_bn)});
            auto tensorO = make_tensor<2>(
                buffers->score,
                {static_cast<size_t>(cur_BM), static_cast<size_t>(gemm_bn)});
            matmul.compute_async<MatmulBlockConfig::M128_N32>(
                tensorP, tensorV, true);
            matmul.storeC_spm_async(tensorO);
            matmul.wait_storeC(1);

            for (int ai = 0; ai < cur_BM; ai++) {
                float *accum_row =
                    buffers->out_accum + ai * BK + hb * gemm_bn;
                float *score_row = buffers->score + ai * gemm_bn;
                int ak = 0;
                for (; ak + 16 <= gemm_bn; ak += 16) {
                    floatv16 accum;
                    floatv16 value;
                    simd_load(accum, accum_row + ak);
                    simd_load(value, score_row + ak);
                    accum = simd_add(accum, value);
                    simd_store(accum, accum_row + ak);
                }
                for (; ak < gemm_bn; ak++) {
                    accum_row[ak] += score_row[ak];
                }
            }
        }

        float *tmp = l_ptr;
        l_ptr = l_next;
        l_next = tmp;
        tmp = m_ptr;
        m_ptr = m_next;
        m_next = tmp;
    }

    for (int row = 0; row < cur_BM; row++) {
        float l_inv = l_ptr[row] > 0.0f ? 1.0f / l_ptr[row] : 0.0f;
        floatv16 vl_inv = l_inv;
        float *out_row = buffers->out_accum + row * args.size_per_head;
        half *out_half_row = buffers->out + row * args.size_per_head;
        int k = 0;
        for (; k + 16 <= args.size_per_head; k += 16) {
            floatv16 vf;
            halfv16 vh;
            simd_load(vf, out_row + k);
            vf = vf * vl_inv;
            vh = (halfv16)vf;
            simd_store(vh, out_half_row + k);
        }
        for (; k < args.size_per_head; k++) {
            out_half_row[k] = (half)(out_row[k] * l_inv);
        }
    }

    size_t out_dst_stride = (args.local_head_num - 1) * token_size;
    half *out_base = (half *)args.fmha_out +  // NOLINT(readability/casting)
                     (q_base + q_offset) *
                         args.local_head_num * args.size_per_head +
                     head_idx * args.size_per_head;
    memcpy_stride(out_base, buffers->out, cur_BM * token_size, token_size,
                  {out_dst_stride, 0});
}


#endif  // TECO_UAL_KERNEL_FLASH_ATTENTION_FLASH_ATTENTION_COMMON_H_
