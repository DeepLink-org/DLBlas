/**
 * Standalone program to pre-compute TCubeTiling constants for MTPBlock kernels.
 *
 * Build: g++ -std=c++17 -I$ASCEND_HOME_PATH/aarch64-linux/include \
 *        -I$ASCEND_HOME_PATH/aarch64-linux/asc/include \
 *        gen_tiling.cpp -o gen_tiling \
 *        -L$ASCEND_HOME_PATH/lib64 -ltiling_api -lregister -lplatform \
 *        -lgraph_base -lunified_dlog -lc_sec -ldl
 *
 * Run:  LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64 ./gen_tiling
 */

#include <cstdio>
#include <cstring>
#include <cstdint>

#include "adv_api/matmul/matmul_tiling.h"
#include "adv_api/matmul/matmul_tiling_base.h"

using namespace matmul_tiling;
using namespace platform_ascendc;

static void print_tiling(const char* name, uint32_t M, uint32_t N, uint32_t K,
                          bool isTransA, bool isTransB)
{
    PlatformInfo platformInfo;
    platformInfo.socVersion = SocVersion::ASCEND910B;
    platformInfo.l1Size   = 524288;
    platformInfo.l0CSize  = 131072;
    platformInfo.ubSize   = 196608;
    platformInfo.l0ASize  = 65536;
    platformInfo.l0BSize  = 65536;

    MatmulApiTiling tiling(platformInfo);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, isTransA);
    tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, isTransB);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    tiling.SetBias(false);
    tiling.SetOrgShape((int32_t)M, (int32_t)N, (int32_t)K);
    tiling.SetShape((int32_t)M, (int32_t)N, (int32_t)K);
    tiling.SetBatchNum(1);

    optiling::TCubeTiling cubeTiling;
    int64_t ret = tiling.GetTiling(cubeTiling);
    if (ret != 0) {
        printf("// ERROR: GetTiling failed for %s\n", name);
        return;
    }

    // Print as C array initializer
    printf("// %s: M=%u N=%u K=%u transA=%d transB=%d\n", name, M, N, K, isTransA, isTransB);
    printf("static const uint32_t tiling_%s_data[] = {\n", name);
    const uint32_t* data = (const uint32_t*)&cubeTiling;
    size_t nwords = sizeof(optiling::TCubeTiling) / sizeof(uint32_t);
    for (size_t i = 0; i < nwords; i++) {
        printf("    0x%08x,%s", data[i], (i % 4 == 3) ? "\n" : " ");
    }
    printf("};\n");
    printf("static const uint32_t tiling_%s_size = %zu;\n\n", name, sizeof(optiling::TCubeTiling));
}

int main()
{
    printf("// Auto-generated TCubeTiling constants for MTPBlock\n");
    printf("// Generated for DAV_2201 / Ascend910B2\n\n");

    // K1: e_proj [b*s, d] × [d, d]^T  → [8, 512] × [512, 512]
    print_tiling("k1_eproj", 8, 512, 512, false, true);

    // K1: h_proj [b*s*hc, d] × [d, d]^T  → [32, 512] × [512, 512]
    print_tiling("k1_hproj", 32, 512, 512, false, true);

    // K2: hc_fn [b*s, hc*d] × [hc*d, mix_hc]^T  → [8, 24] × [2048, 24]
    print_tiling("k2_hcfn", 8, 24, 2048, false, true);

    // K3: wq_a [s, d] × [d, q_lora]^T  → [8, 256] × [512, 256]
    print_tiling("k3_wqa", 8, 256, 512, false, true);

    // K3: wq_b [s*n_heads, q_lora] × [q_lora, head_dim]^T  → [64, 64] × [256, 64]
    print_tiling("k3_wqb", 64, 64, 256, false, true);

    // K3: wkv [s, d] × [d, head_dim]^T  → [8, 64] × [512, 64]
    print_tiling("k3_wkv", 8, 64, 512, false, true);

    // K3: wo_b [s, d] × [d, ng*ol]^T  → [8, 256] × [512, 256]
    print_tiling("k3_wob", 8, 256, 512, false, true);

    // K5: w1/w2/w3 [b*s, d] × [d, inter]^T  → [8, 512] × [512, 512]
    print_tiling("k5_w1", 8, 512, 512, false, true);
    print_tiling("k5_w2", 8, 512, 512, false, true);
    print_tiling("k5_w3", 8, 512, 512, false, true);

    // K6: hc_head [b*s, hc*d] × [hc*d, hc]^T  → [8, 4] × [2048, 4]
    print_tiling("k6_hchd", 8, 4, 2048, false, true);

    // K6: lm_head [1, d] × [d, vocab]^T  → [1, 1000] × [512, 1000]
    print_tiling("k6_head", 1, 1000, 512, false, true);

    return 0;
}
