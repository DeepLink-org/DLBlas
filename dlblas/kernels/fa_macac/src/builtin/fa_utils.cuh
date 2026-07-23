#pragma once
// fa_utils.cuh — flash-attn fwd helper functions (reauthored, builtin).
// Only depends on the OFFICIAL system cute/mctlass (via $MACA_PATH).
// Thin wrappers over cute primitives + MACA compiler builtins; bodies mirrored
// from the reference (Tri Dao flash_attn utils.h). namespace flash kept so the
// kernel body (my_compute.cuh) is unchanged.

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/clear.hpp>
#include <mctlass/mctlass.h>
#include <mctlass/numeric_types.h>
#include <mctlass/numeric_conversion.h>
#include <cstdint>

namespace flash {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction operators + warp (cross-lane) reducers. Use __shfl_xor_sync.

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x > y ? x : y; }
};
template <>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
    static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
template <>
struct Allreduce<2> {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, 1));
        return x;
    }
};

// reduce val(tidx) val(tidx+16) val(tidx+32) val(tidx+48)
struct Partialreduce {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        auto x1 = __shfl_xor_sync(uint64_t(-1), x, 48);
        auto x2 = __shfl_xor_sync(uint64_t(-1), x, 32);
        auto x3 = __shfl_xor_sync(uint64_t(-1), x, 16);
        return op(op(op(x, x1), x2), x3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Partialreduce::run(src(i), op);
    }
}

template <typename Engine0, typename Layout0>
__device__ __forceinline__ void quadreduce_sum(Tensor<Engine0, Layout0> &sum) {
    SumOp<float> sum_op;
    quad_allreduce_(sum, sum, sum_op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Layout reshape: (MMA=4, MMA_M, MMA_N) -> (nrow=(1, MMA_M), ncol=(4, MMA_N))

template <class Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)),
                       make_layout(get<0>(acc_layout), get<2>(acc_layout)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Apply exp2 to all elements (online-softmax numerics: exp2(x*log2e - max*scale)).

template <bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    static_assert(decltype(size<1>(tensor))::value % 2 == 0);
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 scale_vec = {scale, scale};
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        Float2 max_scale_vec = {-max_scaled, -max_scaled};
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni += 2) {
            Float2 x_vec = {tensor(mi, ni), tensor(mi, ni + 1)};
            x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, max_scale_vec);
            tensor(mi, ni) = __builtin_exp2f(x_vec[0]);
            tensor(mi, ni + 1) = __builtin_exp2f(x_vec[1]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ void clear(T &&t) { cute::clear(t); }

template <int N = 2>
__forceinline__ __device__ void sync_threads() {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_ex(N);
}
template <int N = 2>
__forceinline__ __device__ void barrier() {
    __builtin_mxc_barrier_ex(N);
}
template <int M = 0, int N = 2>
__forceinline__ __device__ void barrier_gvm() {
    __builtin_mxc_arrive_gvmcnt(M);
    __builtin_mxc_barrier_ex(N);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ dim3 get_bidInfo(const int &blockType) {
    int m_block = blockIdx.y;
    int bidb = blockIdx.z;
    int bidh = blockIdx.x;
    if (blockType == 0) { int m_block = blockIdx.x; int bidb = blockIdx.z; int bidh = blockIdx.y; return dim3(m_block, bidb, bidh); }
    if (blockType == 1) { int m_block = blockIdx.x; int bidb = blockIdx.y; int bidh = blockIdx.z; return dim3(m_block, bidb, bidh); }
    if (blockType == 2) { int m_block = blockIdx.y; int bidb = blockIdx.z; int bidh = blockIdx.x; return dim3(m_block, bidb, bidh); }
    if (blockType == 3) { int m_block = blockIdx.y; int bidb = blockIdx.x; int bidh = blockIdx.z; return dim3(m_block, bidb, bidh); }
    if (blockType == 4) { int m_block = blockIdx.z; int bidb = blockIdx.x; int bidh = blockIdx.y; return dim3(m_block, bidb, bidh); }
    if (blockType == 5) { int m_block = blockIdx.z; int bidb = blockIdx.y; int bidh = blockIdx.x; return dim3(m_block, bidb, bidh); }
    return dim3(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp32 accumulator -> half register tensor conversion (mctlass NumericArrayConverter).

#define CONVERT_TENSOR_TYPE(type_s, type_d, tensor_s, tensor_d)                                                                                                        \
    constexpr int tensor_d##_numel = decltype(size(tensor_s))::value;                                                                                                   \
    mctlass::NumericArrayConverter<type_d, type_s, tensor_d##_numel> tensor_d##_convert_op;                                                                            \
    auto tensor_d##_frag = tensor_d##_convert_op(*reinterpret_cast<const mctlass::Array<type_s, tensor_d##_numel> *>(tensor_s.data()));                               \
    Tensor tensor_d = make_tensor(make_rmem_ptr<type_d>(&tensor_d##_frag), tensor_s.layout());

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tiled MMA wrappers (QK^T and PV). Pure cute::gemm + cute::copy + retile_D.

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                            Tensor4 const &tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int k = 0; k < size<2>(tCrB); ++k) {
        #pragma unroll
        for (int n = 0; n < size<1>(tCrB); n++) {
            #pragma unroll
            for (int m = 0; m < size<1>(tCrA); m++) {
                cute::gemm(tiled_mma, tCrA(_, m, k), tCrB(_, n, k), acc(_, m, n));
            }
            if (k < size<2>(tCrB) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, n, k + 1), tCrB_copy_view(_, n, k + 1));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// K reg-staging copies (global->reg, reg->smem). Even-MN/K path (our workload).

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void copy_global_to_reg(Tensor<Engine0, Layout0> const &S,
                            uint32_t *D_ptr, Tensor<Engine1, Layout1> const &identity_MN,
                            const int &d, const int &max_MN = 0) {
    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            const int idx = m * size<2>(S) * 4 + k * 4;
            auto src_ptr = (VecType *)(S(_, m, k).data().ptr_);
            auto dst_ptr = (VecType *)(D_ptr + idx);
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            if constexpr (Is_even_MN && Is_even_K) {
                dst_ptr[0] = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                dst_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                                col_mask && row_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_reg_to_share(uint32_t *S_ptr, Tensor<Engine0, Layout0> &D) {
    #pragma unroll
    for (int m = 0; m < size<1>(D); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(D); ++k) {
            const int idx = m * size<2>(D) * 4 + k * 4;
            cute::copy_reg_to_share(S_ptr + idx, D(_, m, k));
        }
    }
}

} // namespace flash
