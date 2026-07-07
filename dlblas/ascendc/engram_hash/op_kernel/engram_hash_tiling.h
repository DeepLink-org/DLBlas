/**
 * engram_hash — Tiling data structure (shared by kernel + host + torch)
 * ASCEND910B2 / DAV_2201
 *
 * N-gram embedding index hash operator.
 *   ngram_token_ids[NT, N]     int32
 *   multipliers[L, N]          int64
 *   vocab_sizes[L, N-1, T]     int32
 *   offsets[L, W]              int32   (W = (N-1)*T)
 * → output[L, NT, W]           int32
 *
 * Algorithm (per (layer, token)):
 *   h = int64(ngram[tk,0]) * mult[l,0]
 *   for i in 1..N-1:
 *     h ^= int64(ngram[tk,i]) * mult[l,i]     (prefix XOR chain)
 *     for t in 0..T-1:
 *       col = (i-1)*T + t
 *       out[l,tk,col] = int32(h % vocab[l,i-1,t]) + off[l,col]
 */
#ifndef ENGRAM_HASH_TILING_H
#define ENGRAM_HASH_TILING_H

#include <cstdint>

// DAV_2201 usable UB budget (192KB total; reserve for stack/alignment margin).
constexpr uint32_t EH_UB_AVAIL = 184 * 1024;
// Margin reserved on top of resident tables when sizing token tiles.
constexpr uint32_t EH_UB_MARGIN = 8 * 1024;

struct EngramHashTilingData {
    // ── problem sizes ──
    uint32_t numTokens;      // NT
    uint32_t ngramSize;      // N  (= max_ngram_size)
    uint32_t numLayers;      // L
    uint32_t numTables;      // T
    uint32_t ngramPos;       // P  = N-1
    uint32_t outWidth;       // W  = (N-1)*T
    // ── multi-core split (token dim) ──
    uint32_t tokensPerCore;  // tokens handled by each non-tail core
    uint32_t tailTokens;     // tokens handled by the tail core
    uint32_t blockNum;       // actual number of launched cores
    // ── intra-core tile (UB split) ──
    uint32_t tileTokens;     // tokens processed per tile (8-aligned when possible)
    uint32_t lastTileTokens; // tokens in the last tile of a full (tokensPerCore) core
    // ── alignment flags (informational) ──
    uint32_t inAligned;      // ngram tile 32B aligned
    uint32_t outAligned;     // output segment 32B aligned
};

#endif  // ENGRAM_HASH_TILING_H
