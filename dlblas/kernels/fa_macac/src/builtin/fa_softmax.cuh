#pragma once
// fa_softmax.cuh — online-softmax (fwd) reauthored, builtin.
// Only depends on the OFFICIAL system cute/mctlass (via $MACA_PATH) +
// our fa_utils.cuh. Non-dropout / non-sink path only (our workload is
// fp16/non-causal/no-dropout/even-MN/K). Bodies mirrored from the reference
// (Tri Dao flash_attn softmax.h). namespace flash kept so the kernel body is unchanged.

#include "fa_utils.cuh"

namespace flash {

template <int kNRows>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {}

    // softmax_rescale_o: online row-max rescale of acc_o + exp2 + row-sum.
    // 3-arg overload (no smem sRowMax sharing) — the one our kernel uses.
    template <bool Is_first, bool Check_inf = false, bool Syncthreads = false, bool AddVec = false,
              typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        MaxOp<float> max_op;
        static_assert(decltype(size<0>(scores))::value == kNRows);
        static_assert(decltype(size<1>(scores))::value % 2 == 0);
        typedef __NATIVE_VECTOR__(2, float) Float2;
        if constexpr (Is_first) {
            flash::template thread_reduce_</*zero_init=*/true>(scores, row_max, max_op);
            if (Syncthreads) flash::sync_threads();
            flash::template quad_allreduce_(row_max, row_max, max_op);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            if constexpr (AddVec) {
                #pragma unroll
                for (int mi = 0; mi < size<0>(scores); mi++) {
                    Float2 x_vec = {0.0f, 0.0f};
                    Float2 scale_vec = {1.0f, 1.0f};
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ni += 2) {
                        Float2 beta_vec = {scores(mi, ni), scores(mi, ni + 1)};
                        x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                    }
                    row_sum(mi) = x_vec[0] + x_vec[1];
                }
            } else {
                SumOp<float> sum_op;
                flash::thread_reduce_</*zero_init=*/true>(scores, row_sum, sum_op);
            }
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template thread_reduce_</*zero_init=*/false>(scores, row_max, max_op);
            if (Syncthreads) flash::sync_threads();
            flash::template quad_allreduce_(row_max, row_max, max_op);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            static_assert(decltype(size<1>(acc_o_rowcol))::value % 2 == 0);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = __builtin_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                Float2 scale_vec = {scores_scale, scores_scale};
                Float2 beta_vec = {0.0f, 0.0f};
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ni += 2) {
                    Float2 x_vec = {acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni + 1)};
                    x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                    acc_o_rowcol(mi, ni) = x_vec[0];
                    acc_o_rowcol(mi, ni + 1) = x_vec[1];
                }
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); mi++) {
                if constexpr (AddVec) {
                    Float2 x_vec = {row_sum(mi), 0.0f};
                    Float2 scale_vec = {1.0f, 1.0f};
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ni += 2) {
                        Float2 beta_vec = {scores(mi, ni), scores(mi, ni + 1)};
                        x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                    }
                    row_sum(mi) = x_vec[0] + x_vec[1];
                } else {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ni++) {
                        row_sum(mi) += scores(mi, ni);
                    }
                }
            }
        }
    }

    // normalize: acc_o /= row_sum, return log-sum-exp = row_max*scale + log(sum).
    template <bool Is_dropout = false, bool Split = false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout = 1.0) {
        flash::quadreduce_sum(row_sum);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        static_assert(decltype(size<1>(acc_o_rowcol))::value % 2 == 0);
        typedef __NATIVE_VECTOR__(2, float) Float2;
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            Float2 scale_vec = {scale, scale};
            Float2 beta_vec = {0.0f, 0.0f};
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ni += 2) {
                Float2 x_vec = {acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni + 1)};
                x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                acc_o_rowcol(mi, ni) = x_vec[0];
                acc_o_rowcol(mi, ni + 1) = x_vec[1];
            }
        }
        return lse;
    }
};

} // namespace flash
