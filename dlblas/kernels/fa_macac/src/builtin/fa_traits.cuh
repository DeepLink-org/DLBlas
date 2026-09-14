#pragma once
// fa_traits.cuh — Flash-attn fwd kernel traits (reauthored, builtin).
// Only depends on the OFFICIAL system cute/mctlass (via $MACA_PATH).
// Concrete config mirrored from the reference (Tri Dao flash_attn
// kernel_traits.h) for: kHeadDim=128,kBlockM=128,kBlockN=64,kNWarps=4,
// Is_Q_in_regs=true, Share_Q_K_smem=true, elem=half_t, kHeadDimV=128.
// Template signature preserved so my_compute.cuh's `using MyTraits =
// Flash_fwd_kernel_traits<128,128,64,4,true,true,mctlass::half_t,128>` is unchanged.

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <mctlass/mctlass.h>
#include <mctlass/numeric_types.h>
#include <cstdint>

using namespace cute;

namespace myflash {
namespace xcore1000 {

// Base traits (mirrors Flash_kernel_traits).
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = mctlass::half_t>
struct Flash_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = uint64_t;

    static constexpr bool Has_cp_async = false;

    using MMA_Atom_16x16x16 = std::conditional_t<
        std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x16_F32F16F16F32>,
        MMA_Atom<MACA_16x16x16_F32BF16BF16F32>>;
    using MMA_Atom_16x16x32 = std::conditional_t<
        std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x32_F32F16F16F32>,
        MMA_Atom<MACA_16x16x32_F32BF16BF16F32>>;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;

    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
    using UniversalCopyAtomB32 = Copy_Atom<UniversalCopy<uint32_t>, elem_type>;
    using UniversalCopyAtomB64 = Copy_Atom<UniversalCopy<uint64_t>, elem_type>;
    using UniversalCopyAtomB128 = Copy_Atom<UniversalCopy<uint128_t>, elem_type>;
    using LDSB64Trans4x16Atom = Copy_Atom<Copy_Traits<MACA_LDS_TRANS_4X16>, elem_type>;
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_ = false,
         bool Share_Q_K_smem_ = false, typename elem_type = mctlass::half_t,
         int kHeadDimV_ = kHeadDim_, typename Base = Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using UniversalCopyAtomB32 = typename Base::UniversalCopyAtomB32;
    using UniversalCopyAtomB64 = typename Base::UniversalCopyAtomB64;
    using UniversalCopyAtomB128 = typename Base::UniversalCopyAtomB128;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    using LDSB64Trans4x16Atom = typename Base::LDSB64Trans4x16Atom;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : 32;

    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int MBase = 3;
    static constexpr int SShift = 3;
    static constexpr int SShift_OPT = kBlockKSmem == 32 ? 3 : 4;
    static constexpr int LDSTRANSBSizzle = kBlockKSmem == 32 ? 1 : 2;
    static constexpr int Num_Stages = (kHeadDim == 128 || kBlockKSmem == 32) ? 2 : 1;
    static constexpr int kAtomLayoutMS = std::min(kBlockM / 16, kNWarps);
    static constexpr int kAtomLayoutMO = kAtomLayoutMS;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_16x16x16,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemV>, Int<kBlockN>>,
                                                      Stride<_1, Int<kBlockKSmemV>>>;
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{}, Shape<Int<kHeadDimV>, Int<kBlockN>>{}));

    using SmemLayoutVtNoSwizzle = decltype(tile_to_shape(
        Layout<Shape<_16, Int<kBlockKSmemV>>, Stride<Int<kBlockKSmemV>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmemV>>,
                           Stride<Int<kBlockKSmemV>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDimV>>{}));

    using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint64_t>, Element>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutV{}) * sizeof(Element); // K layout == KV atom
    static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKSize + kSmemVSize;
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
    static constexpr int kRegSize = kSmemSize / sizeof(uint32_t) / kNThreads;

    // Gmem tiled copies (gmem<->smem).
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static constexpr int kGmemThreadsPerRowV = kBlockKSmemV / kGmemElemsPerLoad;
    using GmemLayoutAtomB128 = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                      Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomV = Layout<Shape<Int<kNThreads / kGmemThreadsPerRowV>, Int<kGmemThreadsPerRowV>>,
                                   Stride<Int<kGmemThreadsPerRowV>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(UniversalCopyAtomB128{}, GmemLayoutAtomB128{}, Layout<Shape<_1, _8>>{}));
    using GmemTiledCopyO = decltype(
        make_tiled_copy(UniversalCopyAtomB128{}, GmemLayoutAtomV{}, Layout<Shape<_1, _8>>{}));
};

} // namespace xcore1000
} // namespace myflash
