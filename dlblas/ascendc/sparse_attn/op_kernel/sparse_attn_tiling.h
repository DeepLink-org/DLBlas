/**
 * sparse_attn_tiling.h - Tiling data structure and decision logic
 * Shared between Host and Device (Kernel)
 *
 * Pure C/C++ header, no __aicore__ or __gm__ keywords.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

// ============================================================================
// Compile-time constants
// ============================================================================

constexpr uint32_t UB_BUDGET = 192 * 1024;  // DAV_2201 UB size (bytes)

// Minimum alignment requirements
constexpr uint32_t BF16_ALIGN_ELEMS = 16;    // 16 * 2B = 32B alignment for DataCopyPad
constexpr uint32_t FP32_ALIGN_ELEMS = 8;     // 8 * 4B = 32B alignment for fp32 vector ops

// ============================================================================
// Tiling data structure (passed from Host to Device via GM)
// ============================================================================

struct SparseAttnTilingData {
    // === Input shape dimensions ===
    uint32_t batchSize;        // b
    uint32_t seqLen;           // m
    uint32_t nHeads;           // h
    uint32_t headDim;          // d
    uint32_t topk;             // topk
    uint32_t kvLen;            // n

    // === Multi-core task distribution ===
    uint32_t totalTasks;       // b * m
    uint32_t usedCoreNum;      // min(aicCoreNum, totalTasks)

    // === Chunk dimensions (UB capacity constrained) ===
    uint32_t hChunk;           // H_chunk: heads processed per inner loop iteration
    uint32_t tChunk;           // T_chunk: topk positions per inner loop iteration

    // === Alignment parameters ===
    uint32_t dAligned;         // AlignUp(d, 8) for fp32 vector reduction
    uint32_t dAlignedBf16;     // AlignUp(d, 16) for bf16 DataCopyPad 32B alignment

    // === Loop counts ===
    uint32_t hLoopCount;       // ceil(h / H_chunk)
    uint32_t tLoopCount;       // ceil(topk / T_chunk)

    // === Scalars ===
    float softmaxScale;        // head_dim ** -0.5

    // === Hardware parameter (for debug) ===
    uint32_t ubSize;
};

// ============================================================================
// Tiling decision algorithm
// ============================================================================

inline uint32_t AlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

/**
 * Compute Tiling parameters based on input shape and UB budget.
 *
 * Decision algorithm (from DESIGN.md Section 4.4):
 *   Step 1: Try full-load (H_chunk = h, T_chunk = topk)
 *   Step 2: If UB overflows, compress T_chunk
 *   Step 3: If still overflows, compress H_chunk
 *   Step 4: Ensure minimum viable chunk sizes
 */
inline SparseAttnTilingData ComputeSparseAttnTiling(
    uint32_t b, uint32_t m, uint32_t h, uint32_t d,
    uint32_t topk, uint32_t n, float softmaxScale,
    uint32_t ubSize, uint32_t aicCoreNum)
{
    SparseAttnTilingData td;

    // Fill input dimensions
    td.batchSize = b;
    td.seqLen = m;
    td.nHeads = h;
    td.headDim = d;
    td.topk = topk;
    td.kvLen = n;
    td.softmaxScale = softmaxScale;
    td.ubSize = ubSize;

    // Compute alignment
    td.dAligned = AlignUp(d, FP32_ALIGN_ELEMS);
    td.dAlignedBf16 = AlignUp(d, BF16_ALIGN_ELEMS);

    // Multi-core task distribution
    td.totalTasks = b * m;
    td.usedCoreNum = std::min(aicCoreNum, td.totalTasks);
    if (td.usedCoreNum == 0) td.usedCoreNum = 1;

    // Chunk size decision
    uint32_t H_chunk = h;
    uint32_t T_chunk = topk;
    uint32_t dAligned = td.dAligned;
    uint32_t dAlignedBf16 = td.dAlignedBf16;

    // UB estimation formula (from DESIGN.md Section 4.3):
    //   q_ub:   H_chunk * d * 2         (bf16 input)
    //   kv_ub:  T_chunk * dAlignedBf16 * 2 (bf16 gather buffer)
    //   scores_ub: H_chunk * T_chunk * 4 (fp32)
    //   output_acc: H_chunk * d * 4     (fp32)
    //   tmp_ub: H_chunk * T_chunk * 4   (fp32, softmax workspace)
    //   max_state: H_chunk * 4           (fp32)
    //   sum_state: H_chunk * 4           (fp32)
    //   topk_idxs: topk * 4              (int32)
    //   attn_sink: H_chunk * 4           (fp32)
    //   valid_mask: T_chunk * 1          (uint8)
    //
    // Total ≈ H_chunk*d*2 + T_chunk*dAlignedBf16*2 + H_chunk*T_chunk*8 + H_chunk*d*4 + H_chunk*8 + topk*4 + T_chunk

    auto computeUB = [&](uint32_t hc, uint32_t tc) -> uint32_t {
        uint32_t q_ub       = hc * d * 2;
        uint32_t kv_ub      = tc * dAlignedBf16 * 2;
        uint32_t scores_ub  = hc * tc * 4;
        uint32_t oacc_ub    = hc * d * 4;
        uint32_t tmp_ub     = hc * tc * 4;
        uint32_t max_ub     = hc * 4;
        uint32_t sum_ub     = hc * 4;
        uint32_t idxs_ub    = topk * 4;
        uint32_t sink_ub    = hc * 4;
        uint32_t mask_ub    = tc * 1;
        return q_ub + kv_ub + scores_ub + oacc_ub + tmp_ub
               + max_ub + sum_ub + idxs_ub + sink_ub + mask_ub;
    };

    // Step 1: Try full-load
    if (computeUB(H_chunk, T_chunk) <= UB_BUDGET) {
        // Full-load mode: everything fits
        td.hChunk = H_chunk;
        td.tChunk = T_chunk;
    } else {
        // Step 2: Try compressing T_chunk first
        // Iterate to find valid T_chunk
        bool found = false;
        for (uint32_t tc = topk; tc >= 1; --tc) {
            if (computeUB(h, tc) <= UB_BUDGET) {
                T_chunk = tc;
                H_chunk = h;
                found = true;
                break;
            }
        }

        // Step 3: If T_chunk compression not enough, compress H_chunk too
        if (!found) {
            // Start with small chunk and search
            for (uint32_t hc = 1; hc <= h; ++hc) {
                for (uint32_t tc = 1; tc <= topk; ++tc) {
                    if (computeUB(hc, tc) <= UB_BUDGET) {
                        H_chunk = hc;
                        T_chunk = tc;
                        found = true;
                        goto done;
                    }
                }
            }
        }
        done:
        if (!found) {
            // Minimum viable chunk: 1 head, 1 topk element
            H_chunk = 1;
            T_chunk = 1;
        }
        td.hChunk = H_chunk;
        td.tChunk = T_chunk;
    }

    // Compute loop counts
    td.hLoopCount = CeilDiv(h, td.hChunk);
    td.tLoopCount = CeilDiv(topk, td.tChunk);

    return td;
}
