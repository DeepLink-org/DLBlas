#pragma once
// fa_params.h — Flash attention forward params (reauthored, builtin).
// Plain data struct; field layout mirrors the reference (Tri Dao flash_attn)
// so our kernel body is unchanged. Only includes <cstdint> — no external deps.
// Namespace mcFlashAttn kept so the harness (fa_my.cu) is unchanged.

#include <cstdint>

namespace mcFlashAttn {

struct Qkv_params {
    using index_t = int64_t;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    int h, h_k;
    int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void * __restrict__ p_ptr;
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;
    void * __restrict__ max_logit_ptr;

    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;
    uint32_t ngroups;

    float scale_softmax;
    float scale_softmax_log2;

    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;
    int * __restrict__ seqused_k;
    int *__restrict__ blockmask;

    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    index_t kscale_batch_stride;
    index_t vscale_batch_stride;
    index_t kscale_row_stride;
    index_t vscale_row_stride;
    index_t kscale_head_stride;
    index_t vscale_head_stride;

    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;
    int * __restrict__ cache_batch_idx;
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int dequant_group;
    void *__restrict__ k_scale_ptr;
    void *__restrict__ v_scale_ptr;

    float p_dropout;
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;
    float scale_softmax_rp_dropout;

    int window_size_left, window_size_right;
    float softcap;

    uint64_t rng_state_seed = 0;
    uint64_t rng_state_offset = 0;

    bool is_bf16;
    bool is_causal;
    bool is_seqlens_k_cumulative;
    bool is_rotary_interleaved;

    int num_splits;
    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;
    bool custom_alibi = false;

    bool has_attn_mask;
    void * __restrict__ attn_mask_ptr = nullptr;
    index_t attn_mask_batch_stride = 0;
    index_t attn_mask_nheads_stride = 0;
    index_t attn_mask_row_stride = 0;
    index_t attn_mask_col_stride = 1;
    index_t attn_mask_batch_shape = 1;
    index_t attn_mask_nheads_shape = 1;
    index_t attn_mask_row_shape = 1;
    index_t attn_mask_col_shape = 1;

    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;

    int d_value;
    int d_value_rounded;
    bool is_support_splitkv = false;
    int arch;

    void *__restrict__ s_aux_ptr;
};

} // namespace mcFlashAttn
