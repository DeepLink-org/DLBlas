/**
 * big_fuse PyTorch Extension — Full 3-Kernel Pipeline
 *
 * Launches K0 (bf16→fp32, AIV), K1 (MatMul, AIC), K2 (Post-process, AIV)
 * from within PyTorch using TORCH_LIBRARY.
 *
 * Stream synchronization: stream(true) clears the queue before each launch,
 * ensuring sequential execution of K0 → K1 → K2 with GM intermediate data.
 *
 * NpuArch: DAV_2201 | CANN: 9.0.0
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"            // must be included before torch
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "big_fuse_tiling.h"

// ---- extern declarations for all three kernels ----
// K0: bf16→fp32 conversion (AIV)
extern "C" void big_fuse_k0_kernel(
    uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    void* residualBf16Gm, void* residualFlatFp32Gm, void* tilingGm);

// K1: MatMul (AIC)
extern "C" void big_fuse_k1_kernel(
    uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    void* aFp32GM, void* bFp32GM, void* cFp32GM, void* tilingGM);

// K2: Vector Post-process (AIV)
extern "C" void big_fuse_k2_kernel(
    uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
    void* rFlat, void* rMix, void* rBf16,
    void* pMix, void* cMix, void* lIn, void* tGm);

// ---- Helper ----
static inline int32_t CeilDiv(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

// ---- Compute K0 Tiling ----
static void ComputeK0Tiling(TilingHeaderK0& hdr, int32_t nTokens,
                            int32_t mhcMult, int32_t hiddenSize, int32_t rgs,
                            int32_t vecCoreNum) {
    hdr.nTokens       = nTokens;
    hdr.mhcMult       = mhcMult;
    hdr.hiddenSize    = hiddenSize;
    hdr.rgs           = rgs;
    hdr.tokensPerCore = CeilDiv(nTokens, vecCoreNum);
    hdr.tokensPerTile = 4;
    hdr.vecCoreNum    = vecCoreNum;
    hdr.reserved[0]   = 0;
}

// ---- Compute K2 Tiling ----
static void ComputeK2Tiling(TilingHeaderK2& hdr, int32_t nTokens,
                            int32_t mhcMult, int32_t hiddenSize, int32_t mhcMult3,
                            int32_t rgs, int32_t vecCoreNum,
                            const float* scaleSrc, const float* baseSrc) {
    hdr.nTokens         = nTokens;
    hdr.mhcMult         = mhcMult;
    hdr.hiddenSize      = hiddenSize;
    hdr.mhcMult3        = mhcMult3;
    hdr.rgs             = rgs;
    hdr.tokensPerCore   = CeilDiv(nTokens, vecCoreNum);
    hdr.tokensPerTile   = 2;
    hdr.vecCoreNum      = vecCoreNum;
    hdr.sinkhornRepeat  = 10;
    hdr.rmsEps          = 1e-6f;
    hdr.mhcPreEps       = 1e-6f;
    hdr.mhcSinkhornEps  = 1e-6f;
    hdr.mhcPostMultValue = 1.0f;
    for (int32_t i = 0; i < 4; ++i) hdr.reserved[i] = 0;

    // Expand scale[3] → scaleVec[24]
    for (int32_t i = 0; i < mhcMult; ++i) hdr.scaleVec[i] = scaleSrc[0];
    for (int32_t i = 0; i < mhcMult; ++i) hdr.scaleVec[mhcMult + i] = scaleSrc[1];
    for (int32_t i = 0; i < mhcMult * mhcMult; ++i)
        hdr.scaleVec[2 * mhcMult + i] = scaleSrc[2];
    for (int32_t i = 0; i < mhcMult3; ++i) hdr.baseVec[i] = baseSrc[i];
}

// ---- PyTorch operator implementation ----
namespace ascend_kernel {

std::vector<at::Tensor> big_fuse_torch(
    const at::Tensor& residual,
    const at::Tensor& fn_weight,
    const at::Tensor& mhc_scale,
    const at::Tensor& mhc_base)
{
    // 1. Validate inputs
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bf16");
    TORCH_CHECK(fn_weight.scalar_type() == at::kFloat, "fn_weight must be fp32");
    TORCH_CHECK(mhc_scale.scalar_type() == at::kFloat, "mhc_scale must be fp32");
    TORCH_CHECK(mhc_base.scalar_type() == at::kFloat, "mhc_base must be fp32");
    TORCH_CHECK(residual.is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(fn_weight.is_privateuseone(), "fn_weight must be on NPU");
    TORCH_CHECK(mhc_scale.is_privateuseone(), "mhc_scale must be on NPU");
    TORCH_CHECK(mhc_base.is_privateuseone(), "mhc_base must be on NPU");

    // 2. Extract shape parameters
    auto res_shape = residual.sizes();  // [B, S, M, H]
    TORCH_CHECK(res_shape.size() == 4, "residual must be 4D [B,S,M,H]");
    int32_t B  = static_cast<int32_t>(res_shape[0]);
    int32_t S  = static_cast<int32_t>(res_shape[1]);
    int32_t M4 = static_cast<int32_t>(res_shape[2]);
    int32_t HS = static_cast<int32_t>(res_shape[3]);

    int32_t nTokens  = B * S;
    int32_t RGS      = M4 * HS;
    int32_t N24      = 2 * M4 + M4 * M4;

    TORCH_CHECK(fn_weight.dim() == 2, "fn_weight must be 2D [K, D]");
    TORCH_CHECK(fn_weight.size(0) == N24 && fn_weight.size(1) == RGS,
                "fn_weight shape mismatch");
    TORCH_CHECK(mhc_scale.numel() == 3, "mhc_scale must have 3 elements");
    TORCH_CHECK(mhc_base.numel() == N24, "mhc_base must have K elements");

    // 3. Allocate output tensors (use empty_like pattern to avoid stream issues)
    auto post_mix = at::empty({B, S, M4, 1},
        residual.options().dtype(at::kFloat));
    auto comb_mix = at::empty({B, S, M4, M4},
        residual.options().dtype(at::kFloat));
    auto layer_input = at::empty({B, S, HS},
        residual.options().dtype(at::kBFloat16));

    // 4. Get NPU stream (stream(true) clears queue before returning)
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 5. Get core counts
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t aivCnt = 0, aicCnt = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aivCnt);
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &aicCnt);
    TORCH_CHECK(aivCnt > 0 && aicCnt > 0, "failed to get NPU core count");

    int32_t aivCoreNum = static_cast<int32_t>(aivCnt);
    int32_t aicCoreNum = static_cast<int32_t>(aicCnt);

    // 6. Compute K0 Tiling
    TilingHeaderK0 k0Tiling{};
    ComputeK0Tiling(k0Tiling, nTokens, M4, HS, RGS, aivCoreNum);

    // 7. Compute K1 Tiling (MatMul — simplified, using host pre-computed tiling)
    //    For production, use MatmulApiTiling. Here we use the validated params.
    TilingHeaderK1 k1Tiling{};
    {
        // Use GetTiling from the Ascend C tiling API
        // For now, use the validated parameters from the host code
        // In production, this would call MatmulApiTiling::GetTiling()
        k1Tiling.cubeTiling.M  = nTokens;
        k1Tiling.cubeTiling.N  = N24;
        k1Tiling.cubeTiling.Ka = RGS;
        k1Tiling.cubeTiling.Kb = RGS;
        k1Tiling.cubeTiling.singleCoreM = 64;
        k1Tiling.cubeTiling.singleCoreN = 24;
        k1Tiling.cubeTiling.baseM = 512;
        k1Tiling.cubeTiling.baseN = 32;
        k1Tiling.mTotalCnt  = CeilDiv(nTokens, 64);
        k1Tiling.nTotalCnt  = CeilDiv(N24, 24);
        k1Tiling.totalBlock = k1Tiling.mTotalCnt * k1Tiling.nTotalCnt;
        k1Tiling.mBaseTail  = nTokens - (k1Tiling.mTotalCnt - 1) * 64;
        k1Tiling.nBaseTail  = N24 - (k1Tiling.nTotalCnt - 1) * 24;
        k1Tiling.convTileK  = 0;

        // Ensure usedCoreNum is set
        k1Tiling.cubeTiling.usedCoreNum =
            std::min(k1Tiling.totalBlock, aicCoreNum);
        if (k1Tiling.cubeTiling.usedCoreNum <= 0)
            k1Tiling.cubeTiling.usedCoreNum = 1;
    }

    // 8. Compute K2 Tiling
    int32_t k2CoreNum = aivCoreNum;
    {
        int32_t tpc = CeilDiv(nTokens, k2CoreNum);
        while (k2CoreNum > 1) {
            int32_t lastStart = (k2CoreNum - 1) * tpc;
            if (lastStart < nTokens && tpc % 2 == 0) break;
            k2CoreNum--;
            tpc = CeilDiv(nTokens, k2CoreNum);
        }
    }

    // Copy scale/base to host buffers
    float scaleHost[3], baseHost[24];
    aclrtMemcpy(scaleHost, 3 * sizeof(float),
        mhc_scale.data_ptr(), 3 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(baseHost, N24 * sizeof(float),
        mhc_base.data_ptr(), N24 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    TilingHeaderK2 k2Tiling{};
    ComputeK2Tiling(k2Tiling, nTokens, M4, HS, N24, RGS, k2CoreNum,
                    scaleHost, baseHost);

    // 9. Allocate intermediate GM tensors via ACL
    size_t resFp32B = static_cast<size_t>(nTokens) * RGS * sizeof(float);
    size_t rawB     = static_cast<size_t>(nTokens) * N24 * sizeof(float);
    void* resFp32Dev = nullptr;
    void* rawDev     = nullptr;
    aclrtMalloc(&resFp32Dev, resFp32B, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&rawDev, rawB, ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(resFp32Dev != nullptr && rawDev != nullptr,
                "ACL malloc failed for intermediate tensors");

    // 10. Copy tiling headers to device
    void* tk0Dev = nullptr; void* tk1Dev = nullptr; void* tk2Dev = nullptr;
    aclrtMalloc(&tk0Dev, sizeof(TilingHeaderK0), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&tk1Dev, sizeof(TilingHeaderK1), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&tk2Dev, sizeof(TilingHeaderK2), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tk0Dev, sizeof(TilingHeaderK0), &k0Tiling, sizeof(TilingHeaderK0),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tk1Dev, sizeof(TilingHeaderK1), &k1Tiling, sizeof(TilingHeaderK1),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tk2Dev, sizeof(TilingHeaderK2), &k2Tiling, sizeof(TilingHeaderK2),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // 11. Launch K0: bf16→fp32 conversion (AIV)
    big_fuse_k0_kernel(static_cast<uint32_t>(aivCoreNum), nullptr, aclStream,
        residual.data_ptr(), resFp32Dev, tk0Dev);

    // 12. Launch K1: MatMul (AIC)
    big_fuse_k1_kernel(static_cast<uint32_t>(k1Tiling.cubeTiling.usedCoreNum),
        nullptr, aclStream,
        resFp32Dev, fn_weight.data_ptr(), rawDev, tk1Dev);

    // 13. Launch K2: Vector post-process (AIV)
    big_fuse_k2_kernel(static_cast<uint32_t>(k2CoreNum), nullptr, aclStream,
        resFp32Dev, rawDev,
        residual.data_ptr(),
        post_mix.data_ptr(),
        comb_mix.data_ptr(),
        layer_input.data_ptr(),
        tk2Dev);

    // 14. Synchronize stream (ensure all kernels complete)
    aclrtSynchronizeStream(aclStream);

    // 15. Cleanup intermediate device memory
    aclrtFree(resFp32Dev);
    aclrtFree(rawDev);
    aclrtFree(tk0Dev);
    aclrtFree(tk1Dev);
    aclrtFree(tk2Dev);

    return {post_mix, comb_mix, layer_input};
}

} // namespace ascend_kernel
