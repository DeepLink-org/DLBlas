/**
 * MatmulApiTiling C++ Helper Implementation
 *
 * Uses CANN MatmulApiTiling to compute TCubeTiling for MatmulImpl kernel usage.
 * NpuArch: DAV_2201, CANN 9.0.0
 */

#include "matmul_tiling_helper.h"
#include "adv_api/matmul/matmul_tiling.h"
#include "adv_api/matmul/matmul_tiling_base.h"

#include <cstdint>
#include <cstring>

using namespace matmul_tiling;
using namespace platform_ascendc;

uint32_t ComputeMatmulTiling(
    void* tilingBuf,
    uint32_t bufSize,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool isTransA,
    bool isTransB)
{
    // Validate buffer size (TCubeTiling is typically < 256 bytes)
    if (bufSize < 256) return 0;

    // Fill platform info for DAV_2201 / Ascend910B2
    PlatformInfo platformInfo;
    platformInfo.socVersion = SocVersion::ASCEND910B;
    platformInfo.l1Size   = 524288;   // 512 KB
    platformInfo.l0CSize  = 131072;   // 128 KB
    platformInfo.ubSize   = 196608;   // 192 KB
    platformInfo.l0ASize  = 65536;    // 64 KB (typical)
    platformInfo.l0BSize  = 65536;    // 64 KB (typical)

    // Create MatmulApiTiling with platform info
    MatmulApiTiling tiling(platformInfo);

    // Set data types: A=half (bf16 on DAV_2201), B=half, C=float
    tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, isTransA);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, isTransB);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);

    // Disable bias
    tiling.SetBias(false);

    // Set problem shape
    tiling.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N),
                       static_cast<int32_t>(K));

    // Set single-core shape (same as full for single-core operation)
    tiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N),
                    static_cast<int32_t>(K));

    // No batch
    tiling.SetBatchNum(1);

    // Get the computed tiling
    optiling::TCubeTiling cubeTiling;
    int64_t ret = tiling.GetTiling(cubeTiling);
    if (ret != 0) return 0;

    // Copy to output buffer
    size_t tilingSize = sizeof(optiling::TCubeTiling);
    if (tilingSize > bufSize) return 0;
    std::memcpy(tilingBuf, &cubeTiling, tilingSize);

    return static_cast<uint32_t>(tilingSize);
}
