// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#ifndef TECO_UAL_ARGS_FLASH_ATTENTION_ARGS_H_
#define TECO_UAL_ARGS_FLASH_ATTENTION_ARGS_H_

#include "ual/com/def.h"

using tecoops::ual::common::UALDataType;
namespace tecoops {
namespace ual {
namespace args {

typedef struct FlashAttentionArgs {
    int spe_num;
    int phase;
    int split_num;
    int batch_size;
    int size_per_head;
    int local_head_num;
    int local_kv_head_num;
    int block_size;
    int block_table_dim;
    int max_q_seq_len;
    int max_k_seq_len;
    int max_block_num;
    float softmax_scale;
    const int *q_seq_lens;
    const int *kv_seq_lens;
    const int *block_table;
    const void * q;
    void *key_cache;
    void *value_cache;
    void *fmha_out;
    void *workspace;
    void *kscratch;  // FA-30: GM scratch holding pre-transposed K tiles
                     // ([head_size][block_size] per (physical block, kv
                     // head)). Phase 3 writes it once per call; phase 0
                     // loads K tiles from it with plain memcpy instead of
                     // the serialized hardware transpose engine (which was
                     // re-running the same transpose once per (row-tile,
                     // col-tile) pair). nullptr -> legacy per-pair path.
    int dbg_stub;  // debug: bit0 skip QK gemm, bit1 skip softmax, bit2 skip PV,
                   // bit3 skip K transpose/load, bit4 skip V load, bit5 skip exp
                   // (vexp->vsub), bit6 skip max-reduce, bit7 skip p_score
                   // stores, bit8 skip o_accum rescale, bit9 skip zero-fill
                   // (all timing only)
    int hexp;      // FA-22: fully-valid softmax rows batch their exp through
                   // the half-precision exp128v2 unit (128 halves per call,
                   // ~265 cycles) instead of float vexp (200 cycles per 16
                   // floats): exp time drops ~12x, p_score values are
                   // half-rounded exp (<=~1e-3 rel err, within atol). Host
                   // gates on TECOOPS_FA22_HEXP (default on, "0" off).
    int fa31;      // FA-31: async K memcpy ping-pong on the FA-30 path.
                   // Host gates on TECOOPS_FA31 (default on, "0" off).
    int fa33;      // FA-33: decode direct-SIMD 2-block merged scan. Two
                   // consecutive physical blocks (contiguous block table,
                   // the official single-sequence case) are processed per
                   // iteration: one 64-token online-softmax visit and a
                   // halved per-iteration fixed cost (Q convert, DMA
                   // waits, loop bookkeeping). Non-contiguous tables fall
                   // back to per-block iterations (bit-identical).
                   // Host gates on TECOOPS_FA33 (default on, "0" off).
    int fa40;      // FA-40: decode FA-33 loop with 2-iteration DMA
                   // lookahead (3 buffer sets). The FA-39 probe showed
                   // 33-47% of the decode wall is exposed compute/DMA
                   // stalls with the 1-iteration lookahead; fetching two
                   // iterations ahead gives each DMA ~2 iterations to
                   // complete before its first read. Default off (A/B);
                   // the non-fa40 path is byte-identical to FA-33.
    int fa35;      // FA-35: phase-2 (split reduction) bulk partial fetch.
                   // A task's split_num partials ([split][BK+2], 264B each)
                   // are contiguous in workspace; the legacy loop fetched
                   // one per tiny memcpy (16 ops/task). This fetches them
                   // in 2KB (8-split) chunks with one bulk memcpy each,
                   // cutting the reduction kernel's per-DMA-op cost.
                   // Falls back to per-split memcpy on malloc failure.
                   // Host gates on TECOOPS_FA35 (default on; "0" off).
    int fa47;      // FA-47: prefill 4-page batching (chunked whitelist).
                   // Four KV pages (128 score columns) per softmax visit
                   // halve the visit count, so the per-(row,visit)
                   // max-reduce tree (stub bit16: -12~-15% of prefill, the
                   // biggest chain component) and per-visit fixed costs run
                   // half as often. Shares the FA-28 chain with cols=4BN,
                   // PV gemm K=4BN per head-dim half. SPM: fits only for
                   // tile_blocks==1 (BM-sized row state). Host gates on
                   // TECOOPS_FA47 (default ON; "0" restores the exact
                   // FA-28 2-page behavior).
    int fa44;      // FA-44: prefill o_accum rescale 16-row vectorized
                   // group skip. The rescale loop paid a scalar
                   // load+compare+branch per row per (sub-block, KV-visit);
                   // with the running max stable (scale==1.0f) for most
                   // rows, the pass was pure SPM-roundtrip overhead
                   // (stub-measured -6.7% of prefill, uniform T1-T4).
                   // FA-44 checks 16 scales per floatv16 load+redsum and
                   // skips the group with one branch; rows with scale !=
                   // 1.0f are rescaled with the exact legacy math
                   // (bit-identical: rescale iff scale != 1.0f, and the
                   // grouped test skips exactly the groups whose every
                   // lane is 1.0f). Host gates on TECOOPS_FA44 (default
                   // on, "0" off).
    int fa48x;     // FA-48X (axis-swapped attention, breakthrough bet A):
                   // for the same chunked whitelist shapes as fa47, swap
                   // the score axis so each SIMD lane is one QUERY:
                   //   QK:  S^T = K_native[128k,64] x Q_g^T[64,32] (4
                   //        gemms, one per 32-query group) -> OUT layout
                   //        [key][query]; K is read from the native cache
                   //        (no phase-3 K transpose dependency).
                   //   softmax: per-lane max/exp/sum along the key
                   //        direction -> no 15-lane horizontal max tree.
                   //   PV:   O^T = V^T[64,128k] x P^T_g[128k,32], V^T
                   //        staged in the phase-3 scratch (args.vscratch)
                   //        by the same parallel scalar transpose pass
                   //        that already produces K^T.
                   //   O:    O^T [d][q] accumulator, per-lane online
                   //        rescale, final per-row normalize + transpose.
                   // Host gates on TECOOPS_FA48X (default ON; "0"
                   // restores the exact FA-47 path). Same math, different
                   // association order (fp reassociation only).
    void *vscratch;  // FA-48X: phase-3 V^T scratch (second half of the
                     // FA-30 workspace), layout identical to kscratch:
                     // [block_id * kv_head_num + kv_head_idx][64][32].
    int fa61;      // FA-61: per-group QK<->softmax pipelining inside the
                   // fa48x visit (2 outstanding ax_qk gemms, drain-per-
                   // group wait_storeC(1); the FA-60 K prefetch moves to
                   // the g==3 drain, the only point where every kbuf2
                   // reader is done). Same math and same score slots;
                   // only the issue/wait order changes. Host side sets
                   // this unconditionally (fixed-on production choice;
                   // no environment surface).
    int fa63;      // FA-63: visit-head overlap inside the fa48x visit.
                   // The (b) V^T wait+SIMD-scatter block is moved to after
                   // the first QK gemms are enqueued (ACE) so the scatter
                   // runs under them. Block body byte-identical in both
                   // positions; completes before the first vt48 consumer,
                   // so all consumers see identical bytes at identical
                   // times. Only issue ORDER changes.
    int fa65;      // FA-65 (A1' K-HOST-SUBMIT r3, 20260825): decode split-KV
                   // phase-2 submission elimination. When set AND the fixed
                   // production decode whitelist holds, the phase-1 kernel
                   // tail performs the exact phase-2 combine inline: each
                   // task atomically arrives on its class-separated int64
                   // arrival cell (fa65_ctr) after its partial write; the
                   // task whose RMW return closes the group runs the byte-
                   // identical reduction for that group's heads. No spin,
                   // no poll, no grid barrier; output must stay cross-
                   // bitwise equal to the two-launch path. Host skips the
                   // second run() when set.
    void *fa65_ctr;  // FA-65: GM base of the fixed 4-cell arrival slot
                     // ({spl16 g0, spl16 g1, spl32, pad} int64 cells) that
                     // the host places at float offset 8448 of the decode
                     // workspace (past the production partial region).
                     // nullptr whenever fa65 == 0.
} FlashAttentionArgs;

typedef struct FlashAttentionPatchArgs {
    FlashAttentionArgs *rvargs;
    UALDataType data_type;
} FlashAttentionPatchArgs;

}  // namespace args
}  // namespace ual
}  // namespace tecoops

#endif  // TECO_UAL_ARGS_FLASH_ATTENTION_ARGS_H_
