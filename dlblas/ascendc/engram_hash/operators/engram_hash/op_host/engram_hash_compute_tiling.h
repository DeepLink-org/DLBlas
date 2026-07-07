/**
 * engram_hash — host-side tiling computation (shared by direct-invoke main
 * and the PyTorch extension). Host-only; must NOT include kernel_operator.h.
 */
#ifndef ENGRAM_HASH_COMPUTE_TILING_H
#define ENGRAM_HASH_COMPUTE_TILING_H

#include "engram_hash_tiling.h"
#include <cstdint>

// Compute the full tiling for engram_hash.
//   NT, N, L, T : problem sizes
//   coreNum     : available Vector cores (platform-provided, e.g. 48)
static inline void ComputeEngramHashTiling(
    uint32_t NT, uint32_t N, uint32_t L, uint32_t T,
    uint32_t coreNum, EngramHashTilingData& td)
{
    const uint32_t P = (N >= 1) ? (N - 1) : 0;
    const uint32_t W = P * T;

    td.numTokens = NT;
    td.ngramSize = N;
    td.numLayers = L;
    td.numTables = T;
    td.ngramPos  = P;
    td.outWidth  = W;

    // ── multi-core split on the token dimension ──
    uint32_t usedCores = (coreNum < NT) ? coreNum : NT;
    if (usedCores < 1) usedCores = 1;
    uint32_t tokensPerCore = (NT + usedCores - 1) / usedCores;
    if (tokensPerCore < 1) tokensPerCore = 1;
    uint32_t blockNum = (NT + tokensPerCore - 1) / tokensPerCore;
    if (blockNum < 1) blockNum = 1;
    uint32_t tailTokens = NT - tokensPerCore * (blockNum - 1);

    td.tokensPerCore = tokensPerCore;
    td.tailTokens    = tailTokens;
    td.blockNum      = blockNum;

    // ── intra-core tile budget (UB split) ──
    // resident = multipliers(int64) + vocab(int32) + offsets(int32)
    // perToken = ngram row(int32) + output for all L layers(int32)
    const uint32_t residentUb = L * N * (uint32_t)sizeof(int64_t)
                              + L * W * (uint32_t)sizeof(int32_t)
                              + L * W * (uint32_t)sizeof(int32_t);
    uint32_t perTokenUb = N * (uint32_t)sizeof(int32_t)
                        + L * W * (uint32_t)sizeof(int32_t);
    if (perTokenUb < 1) perTokenUb = 1;

    uint32_t tileTokens = 1;
    if (EH_UB_AVAIL > residentUb + EH_UB_MARGIN) {
        tileTokens = (EH_UB_AVAIL - residentUb - EH_UB_MARGIN) / perTokenUb;
    }
    // Align down to a multiple of 8 (keeps ngram tile / output segments
    // 32B-friendly when N or W already align); clamp to sane bounds.
    tileTokens = (tileTokens / 8) * 8;
    if (tileTokens < 8) tileTokens = 8;
    if (tileTokens > tokensPerCore) tileTokens = tokensPerCore;
    if (tileTokens < 1) tileTokens = 1;

    uint32_t loops = (tokensPerCore + tileTokens - 1) / tileTokens;
    if (loops < 1) loops = 1;
    uint32_t lastTileTokens = tokensPerCore - (loops - 1) * tileTokens;

    td.tileTokens     = tileTokens;
    td.lastTileTokens = lastTileTokens;

    // ── alignment flags (int32: 8 elems = 32B; int64: 4 elems = 32B) ──
    td.inAligned  = ((tileTokens * N) % 8 == 0) ? 1u : 0u;
    td.outAligned = ((tileTokens * W) % 8 == 0) ? 1u : 0u;
}

#endif  // ENGRAM_HASH_COMPUTE_TILING_H
