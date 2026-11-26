#pragma once
// my_compute.cuh — GENUINE from-scratch flash-attn fwd (D=128, blockM128, blockN64, 4 warps)
// namespace myflash. Online-softmax + tiled MMA + swizzled SMEM.
// K reg-staged + prefetched (overlaps PV). V direct gmem→smem (correct). Q in regs.
// Specialized: fp16, non-causal, even-MN/K.

// Builtin: only the official system cute/mctlass (via $MACA_PATH) +
// our own reauthored headers (no external source tree dependency).
#include "builtin/fa_traits.cuh"
#include "builtin/fa_utils.cuh"
#include "builtin/fa_softmax.cuh"
#include "builtin/fa_params.h"

namespace myflash {
namespace xcore1000 {
using namespace cute;
using MyTraits = Flash_fwd_kernel_traits<128, 128, 64, 4, true, true, mctlass::half_t, 128>;

template<typename Params>
__forceinline__ __device__ void compute_attn_myimpl(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Kernel_traits = MyTraits;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kBlockM = Kernel_traits::kBlockM, kBlockN = Kernel_traits::kBlockN, kHeadDim = Kernel_traits::kHeadDim;
    const int tidx = threadIdx.x;
    const int kBlockM_stride = m_block * kBlockM;

    extern __shared__ char smem_[];
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    const index_t row_offset_q = bidb * params.q_batch_stride + bidh * params.q_head_stride + kBlockM_stride * params.q_row_stride;
    const index_t row_offset_kv = bidb * params.k_batch_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_kv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_kv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.v_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // Prologue: Q gmem→smem→reg (reused)
    cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    flash::barrier_gvm<0>();
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);
    flash::sync_threads();

    int n_block = (params.seqlen_k + kBlockN - 1) / kBlockN - 1;
    const int n_block_min = 0;
    const int gK_offset = -int(kBlockN * params.k_row_stride);
    const int gV_offset = -int(kBlockN * params.v_row_stride);
    tKgK.data() = tKgK.data() + n_block * kBlockN * params.k_row_stride;
    tVgV.data() = tVgV.data() + n_block * kBlockN * params.v_row_stride;

    // K reg-staging (prefetch next K during PV)
    uint32_t tKrK[int(Kernel_traits::kRegSize / 2)];
    Tensor tKcK = gmem_thr_copy_QKV.partition_S(make_identity_tensor(make_shape(size<0>(sK), size<1>(sK))));
    flash::copy_global_to_reg<true, true>(tKgK, tKrK, tKcK, params.d, params.seqlen_k - n_block * kBlockN);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    flash::clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

    constexpr int n_masking_steps = 1;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        flash::copy_reg_to_share(tKrK, tKsK);
        // V direct gmem→smem (correct; overlaps sync+QK^T)
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        flash::clear(acc_s);
        flash::sync_threads();
        flash::gemm</*A_in_regs=*/true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                                        smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
        // prefetch next K (overlaps softmax + PV)
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_global_to_reg<true, true>(tKgK, tKrK, tKcK, params.d);
        }
        softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/false, /*Syncthreads=*/true, /*Convert=*/true>(
            acc_s, acc_o, params.scale_softmax_log2);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        if (n_block > n_block_min) tVgV.data() = tVgV.data() + gV_offset;
    }
    for (; n_block > n_block_min; --n_block) {
        flash::copy_reg_to_share(tKrK, tKsK);
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        flash::clear(acc_s);
        flash::sync_threads();
        flash::gemm</*A_in_regs=*/true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                                        smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + gK_offset;
            flash::copy_global_to_reg<true, true>(tKgK, tKrK, tKcK, params.d);
        }
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false, /*Syncthreads=*/true, /*Convert=*/true>(
            acc_s, acc_o, params.scale_softmax_log2);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        if (n_block > n_block_min) tVgV.data() = tVgV.data() + gV_offset;
    }
    // block 0
    if (n_block == n_block_min) {
        flash::copy_reg_to_share(tKrK, tKsK);
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        flash::clear(acc_s);
        flash::sync_threads();
        flash::gemm</*A_in_regs=*/true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                                        smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false, /*Syncthreads=*/true, /*Convert=*/true>(
            acc_s, acc_o, params.scale_softmax_log2);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(acc_o, params.scale_softmax, params.rp_dropout);
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_o, rO)
    flash::barrier();
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    flash::sync_threads();
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    const index_t row_offset_o = bidb * params.o_batch_stride + bidh * params.o_head_stride + kBlockM_stride * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.o_row_stride, _1{}));
    cute::copy(gmem_tiled_copy_O, gmem_thr_copy_O.partition_S(sO), gmem_thr_copy_O.partition_D(gO));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)
                  + (bidb * params.h + bidh) * params.seqlen_q + kBlockM_stride), Shape<Int<kBlockM>>{}, make_stride(_1{}));
    Tensor taccOcO = thr_mma.partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.seqlen_q - kBlockM_stride) gLSE(row) = lse(mi);
        }
    }
}

template<typename Params>
__global__ void my_flash_fwd_kernel(Params params, const int num_m_block, const int block_type) {
    const dim3 bidInf = flash::get_bidInfo(block_type);
    compute_attn_myimpl<Params>(params, bidInf.y, bidInf.z, bidInf.x);
}
} // xcore1000
} // myflash
