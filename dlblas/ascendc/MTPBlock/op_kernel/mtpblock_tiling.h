/**
 * MTPBlock AscendC 算子 - 共享 Tiling 头文件
 *
 * 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字。
 * 所有 kernel 文件和 host 文件共用此头文件。
 *
 * NpuArch: DAV_2201 (arch22)
 * Target: Ascend910B2
 */

#pragma once

#include <cstdint>

// ============================================================================
// 硬件参数 (DAV_2201 / Ascend910B2)
// ============================================================================
constexpr uint32_t UB_SIZE = 196608;      // 192 KB
constexpr uint32_t L0C_SIZE = 131072;     // 128 KB
constexpr uint32_t L1_SIZE = 524288;      // 512 KB
constexpr uint32_t DOUBLE_BUFFER = 2;

// ============================================================================
// Tiling 常量
// ============================================================================
constexpr uint32_t TILE_LENGTH = 4096;     // 单次搬运元素数基准
constexpr uint32_t ALIGN_32B = 32;         // 32 字节对齐

// ============================================================================
// 演示参数 (b=1, s=8, hc=4, d=512)
// ============================================================================
constexpr uint32_t DEMO_B = 1;
constexpr uint32_t DEMO_S = 8;
constexpr uint32_t DEMO_HC = 4;
constexpr uint32_t DEMO_D = 512;
constexpr uint32_t DEMO_VOCAB = 129280;
constexpr uint32_t DEMO_N_HEADS = 8;
constexpr uint32_t DEMO_HEAD_DIM = 64;
constexpr uint32_t DEMO_ROPE_HEAD_DIM = 32;
constexpr uint32_t DEMO_Q_LORA = 256;
constexpr uint32_t DEMO_O_LORA = 128;
constexpr uint32_t DEMO_O_GROUPS = 2;
constexpr uint32_t DEMO_WIN = 8;
constexpr uint32_t DEMO_N_EXPERTS = 8;
constexpr uint32_t DEMO_TOPK = 2;
constexpr uint32_t DEMO_SINKHORN_ITERS = 20;
constexpr uint32_t DEMO_MIX_HC = (2 + DEMO_HC) * DEMO_HC;  // = 24
constexpr float DEMO_EPS = 1e-6f;
constexpr float DEMO_SOFTMAX_SCALE = 0.125f;  // head_dim^-0.5 = 64^-0.5

// ============================================================================
// MatMul 最大数量 (每个 kernel 内嵌的 MatMul 数上限)
// ============================================================================
constexpr uint32_t MAX_MATMUL_PER_KERNEL = 3;

// ============================================================================
// 精简 TCubeTiling - 手动填充, 避免 Host 侧 MatmulApiTiling 依赖
//
// 背景: MatmulApiTiling 需要 C++ 编译 (使用 std::string 等), 无法在 ASC host
//       直调代码中直接调用。Torch extension (C++ 编译) 可通过 matmul_tiling_helper
//       正确计算。ASC host 直调使用手动填充的 tiling 参数。
//
// 字段说明:
//   M, N, Ka, Kb      - 完整问题维度
//   singleCoreM/N/K    - 单核维度 (= 全量, 当 usedCoreNum=1)
//   baseM, baseN, baseK- 单次 Iterate tile 大小 (基于 L0C/L1 容量)
//   usedCoreNum        - 使用的 AI Core 数
//   isBias             - 是否有 bias (0/1)
//   transLength        - transpose 标记
//
// 对于 DAV_2201 (L0C=128KB, float C 矩阵):
//   baseM × baseN × 4 ≤ L0C, baseK ≤ L1/(baseM*2 + baseN*2)
//   常见: baseM=8, baseN=64 用于 fp16 输入 / fp32 输出
// ============================================================================
struct SimpleCubeTiling {
    int32_t M;
    int32_t N;
    int32_t Ka;
    int32_t Kb;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t singleCoreK;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t usedCoreNum;
    int32_t isBias;
    int32_t transLength;
    // Padding to align with TCubeTiling size
    int32_t _pad[28];
};

// ============================================================================
// Kernel 枚举 - 标识当前运行哪个 kernel
// ============================================================================
enum MTPKernelType : uint32_t {
    MTP_K1_EMBED_FUSE = 1,
    MTP_K2_HC_PRE     = 2,
    MTP_K3_ATTN_BLOCK = 3,
    MTP_K4_HC_POST    = 4,
    MTP_K5_MOE_BLOCK  = 5,
    MTP_K6_MTP_HEAD   = 6,
};

// ============================================================================
// 通用 TilingData - 所有 kernel 共用基类结构
// ============================================================================
struct MTPBlockTilingData {
    // Kernel 标识
    MTPKernelType kernelType;

    // Shape 信息
    uint32_t b, s, hc, d;
    uint32_t head_dim, rope_head_dim, n_heads, n_groups;
    uint32_t q_lora, o_lora, vocab_size, win, n_experts, topk;
    uint32_t hc_sinkhorn_iters, mix_hc;

    // Tile 参数
    uint32_t tile_s;         // s 维度 tile 大小
    uint32_t usedCoreNum;    // 使用的核数
    uint32_t blockNum;       // block 数量

    // 各 tensor 的 shape/stride 信息 (元素数)
    uint64_t totalElements;  // 当前 kernel 输入元素总数

    // hc_pre/hc_post 专用
    float hc_eps;
    float norm_eps;
    float softmax_scale;
};

// ============================================================================
// K1: mtp_embed_fuse TilingData
// ============================================================================
struct K1EmbedFuseTiling {
    MTPBlockTilingData base;
    // Embedding 表大小
    uint32_t vocab_size_k1;
    // 输入 tensor 的 GM 偏移 (相对于 base GM 地址)
    uint64_t x_offset;           // input x [b,s,hc,d]
    uint64_t input_ids_offset;   // input_ids [b,s]
    uint64_t embed_weight_offset;// embedding weight [vocab,d]
    uint64_t enorm_weight_offset;// enorm weight [d]
    uint64_t e_proj_weight_offset;
    uint64_t h_proj_weight_offset;
    uint64_t hnorm_weight_offset;
    uint64_t feat_offset;        // output [b,s,hc,d]
};

// ============================================================================
// K2: hc_pre TilingData (共享 kernel, Attn/FFN 各调用一次)
// ============================================================================
struct K2HcPreTiling {
    MTPBlockTilingData base;
    uint64_t x_offset;           // input x [b,s,hc,d]
    uint64_t hc_fn_offset;       // projection weight [mix_hc, hc*d]
    uint64_t hc_scale_offset;    // scale [3]
    uint64_t hc_base_offset;     // base [mix_hc]
    uint64_t y_offset;           // output y [b,s,d]
    uint64_t pre_offset;         // output pre [b,s,hc]
    uint64_t post_offset;        // output post [b,s,hc]
    uint64_t comb_offset;        // output comb [b,s,hc,hc]
    bool is_attn_side;           // true=attn sub-block, false=ffn sub-block
};

// ============================================================================
// K3: attn_block TilingData
// ============================================================================
struct K3AttnBlockTiling {
    MTPBlockTilingData base;
    uint64_t x_offset;           // input x [b,s,d]
    // Q weights
    uint64_t wq_a_weight_offset; // [q_lora, d]
    uint64_t q_norm_weight_offset;
    uint64_t wq_b_weight_offset; // [n_heads*head_dim, q_lora]
    // KV weights
    uint64_t wkv_weight_offset;  // [head_dim, d]
    uint64_t kv_norm_weight_offset;
    // Output weights
    uint64_t wo_a_weight_offset; // [n_groups, o_lora, n_heads*head_dim/n_groups]
    uint64_t wo_b_weight_offset; // [d, n_groups*o_lora]
    // Attention sink
    uint64_t attn_sink_offset;   // [n_heads]
    // RoPE freqs_cis
    uint64_t freqs_cis_offset;   // [s, rope_dim/2] complex64
    // TopK indices (precomputed by host)
    uint64_t topk_idxs_offset;   // [b, s, win] int32
    // Output
    uint64_t attn_out_offset;    // [b,s,d]
};

// ============================================================================
// K4: hc_post TilingData (共享 kernel, Attn/FFN 各调用一次)
// ============================================================================
struct K4HcPostTiling {
    MTPBlockTilingData base;
    uint64_t x_offset;           // input x [b,s,d]
    uint64_t residual_offset;    // input residual [b,s,hc,d]
    uint64_t post_offset;        // input post [b,s,hc]
    uint64_t comb_offset;        // input comb [b,s,hc,hc]
    uint64_t out_offset;         // output [b,s,hc,d]
};

// ============================================================================
// K5: moe_block TilingData
// ============================================================================
struct K5MoeBlockTiling {
    MTPBlockTilingData base;
    uint64_t x_offset;           // input x [b*s, d]
    // Gate weights
    uint64_t gate_weight_offset; // [n_experts, d]
    uint64_t gate_bias_offset;   // [n_experts]
    // Expert weights (每个 expert 的 w1/w2/w3)
    uint64_t expert_w1_offset;   // [n_experts, inter_dim, d]
    uint64_t expert_w2_offset;   // [n_experts, d, inter_dim]
    uint64_t expert_w3_offset;   // [n_experts, inter_dim, d]
    // Shared expert weights
    uint64_t shared_w1_offset;
    uint64_t shared_w2_offset;
    uint64_t shared_w3_offset;
    // Output
    uint64_t ffn_out_offset;     // [b*s, d]
    // Workspace
    uint64_t workspace_offset;   // temp workspace for gate/expert dispatch
    uint32_t inter_dim;
    uint32_t route_scale_int;    // fixed-point scale
    // MatmulImpl tiling (shared expert w1/w2/w3)
    // mm_tiling[0]: w1 [M, K]×[K, N] with B transposed
    // mm_tiling[1]: w3 [M, K]×[K, N] with B transposed
    // mm_tiling[2]: w2 [M, K]×[K, N] with B transposed
    SimpleCubeTiling mm_tiling[MAX_MATMUL_PER_KERNEL];
};

// ============================================================================
// K6: mtp_head TilingData
// ============================================================================
struct K6MtpHeadTiling {
    MTPBlockTilingData base;
    uint64_t x_offset;           // input x [b,s,hc,d]
    uint64_t hc_head_fn_offset;  // [hc, hc*d]
    uint64_t hc_head_scale_offset;
    uint64_t hc_head_base_offset;
    uint64_t norm_weight_offset; // [d]
    uint64_t head_weight_offset; // [vocab, d]
    uint64_t logits_offset;      // output [b, vocab]
};
