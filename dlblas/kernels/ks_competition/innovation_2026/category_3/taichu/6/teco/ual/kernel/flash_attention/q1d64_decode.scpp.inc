// Specialized production decode kernel for batch1/q1/H4/KVH2/D64/BS32,
// split16, FA33+FA40 and FA65. The host launches this symbol only under
// that exact whitelist; every other configuration keeps the canonical kernel.
__global__ void teco_slave_flash_attention_q1d64_split16(
    FlashAttentionArgs args) {
    const int tid = threadIdx;
    if (tid >= 32) return;

    const int BK = 64;
    const int BN = 32;
    const int split_idx = tid & 15;
    const int head_task_idx = tid >> 4;
    const int head_idx = head_task_idx * 2;
    const int kv_head_idx = head_task_idx;
    const int k_seq_len = args.kv_seq_lens[0];
    const int kv_tile_num = (k_seq_len + BN - 1) / BN;
    const int first_tile = kv_tile_num * split_idx / 16;
    const int last_tile = kv_tile_num * (split_idx + 1) / 16;
    const int first_k = first_tile * BN;
    const int last_k = MIN(k_seq_len, last_tile * BN);

    // Allocate only storage consumed by the FA40 decode path. At T4 this is
    // about 56 KiB/SPE including the last-arriver reducer, instead of the
    // monolithic kernel's prefill/decode superset.
    const int table_blocks =
        MIN(args.block_table_dim, args.max_k_seq_len / BN + 2);
    int *ldm_block_table =
        (int *)rt_spm_malloc(table_blocks * sizeof(int));
    memcpy(ldm_block_table, args.block_table,
           table_blocks * sizeof(int));
    half *qbuf = (half *)rt_spm_malloc(2 * BK * sizeof(half));
    memcpy(qbuf, (const half *)args.q + head_idx * BK,
           2 * BK * sizeof(half));

    half *kdual_a = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *kdual_b = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *kdual_alt_a = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *kdual_alt_b = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *kdual3_a = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *kdual3_b = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *vdual_a = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *vdual_b = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *vdual_alt_a = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *vdual_alt_b = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *vdual3_a = (half *)rt_spm_malloc(BN * BK * sizeof(half));
    half *vdual3_b = (half *)rt_spm_malloc(BN * BK * sizeof(half));

    float *outc = (float *)rt_spm_malloc(2 * 2 * BN * sizeof(float));
    float *o_accum = (float *)rt_spm_malloc(2 * BK * sizeof(float));
    float *l_prev = (float *)rt_spm_malloc(2 * sizeof(float));
    float *m_prev = (float *)rt_spm_malloc(2 * sizeof(float));
    float *l_cur = (float *)rt_spm_malloc(2 * sizeof(float));
    float *m_cur = (float *)rt_spm_malloc(2 * sizeof(float));
    float *scale = (float *)rt_spm_malloc(2 * sizeof(float));

    floatv16 vzero = 0.0f;
    for (int h = 0; h < 2 * BK; h += 16)
        simd_store(vzero, o_accum + h);
    l_prev[0] = l_prev[1] = 0.0f;
    l_cur[0] = l_cur[1] = 0.0f;
    fill_inf(m_prev, 2);
    fill_inf(m_cur, 2);
    float *l_ptr = l_prev;
    float *m_ptr = m_prev;
    float *grp_l = l_prev, *grp_ln = l_cur;
    float *grp_m = m_prev, *grp_mn = m_cur;

    MemcpyHandle kd_get_handle[3];
    MemcpyHandle kd2_get_handle[3];
    MemcpyHandle vd_get_handle[3];
    MemcpyHandle vd2_get_handle[3];

                    // FA-40: 2-iteration DMA lookahead for the FA-33 loop.
                    // The FA-39 probe (stub 2048) showed 33-47% of the
                    // decode wall is exposed compute/DMA stalls with the
                    // 1-iteration lookahead (per-iteration DMA latency ~2x
                    // the compute time). Here iteration i reads set i%3 and
                    // issues the fetches for iteration i+2 into set (i+2)%3,
                    // giving each DMA ~2 iteration-times to complete before
                    // its first SPM read. Per-iteration math is identical
                    // to FA-33, so the output is bit-identical; the async
                    // ordering guarantee is the same one FA-33 relies on.
                    half *kseta[3];
                    half *ksetb[3];
                    half *vseta[3];
                    half *vsetb[3];
                    kseta[0] = kdual_a;
                    ksetb[0] = kdual_b;
                    kseta[1] = kdual_alt_a;
                    ksetb[1] = kdual_alt_b;
                    kseta[2] = kdual3_a;
                    ksetb[2] = kdual3_b;
                    vseta[0] = vdual_a;
                    vsetb[0] = vdual_b;
                    vseta[1] = vdual_alt_a;
                    vsetb[1] = vdual_alt_b;
                    vseta[2] = vdual3_a;
                    vsetb[2] = vdual3_b;
                    int jd = 0;
                    int j0 = first_k;
                    float fa39_probe = 0.0f;
                    // pre-loop: iteration 0 sync into set 0, iteration 1
                    // async into set 1.
                    if (j0 < last_k) {
                        int b0 = j0 / args.block_size;
                        int bid0 = ldm_block_table[b0];
                        int nb0 = 1;
                        if (j0 + 2 * BN <= last_k) {
                            int fb1 = (j0 + BN) / args.block_size;
                            if (ldm_block_table[fb1] == bid0 + 1) nb0 = 2;
                        }
                        const half *k_src = (const half *)args.key_cache
                            + bid0 * args.local_kv_head_num *
                                  args.block_size * args.size_per_head
                            + kv_head_idx * args.block_size *
                                  args.size_per_head;
                        const half *v_src = (const half *)args.value_cache
                            + bid0 * args.local_kv_head_num *
                                  args.block_size * args.size_per_head
                            + kv_head_idx * args.block_size *
                                  args.size_per_head;
                        memcpy(kseta[0], k_src,
                               BN * args.size_per_head * sizeof(half));
                        if (nb0 == 2) {
                            memcpy(ksetb[0],
                                   k_src + args.local_kv_head_num *
                                       args.block_size * args.size_per_head,
                                   BN * args.size_per_head * sizeof(half));
                        }
                        memcpy(vseta[0], v_src,
                               BN * args.size_per_head * sizeof(half));
                        if (nb0 == 2) {
                            memcpy(vsetb[0],
                                   v_src + args.local_kv_head_num *
                                       args.block_size * args.size_per_head,
                                   BN * args.size_per_head * sizeof(half));
                        }
                        int j1 = j0 + nb0 * BN;
                        if (j1 < last_k) {
                            int b1 = j1 / args.block_size;
                            int bid1 = ldm_block_table[b1];
                            int nb1 = 1;
                            if (j1 + 2 * BN <= last_k) {
                                int fb1 = (j1 + BN) / args.block_size;
                                if (ldm_block_table[fb1] == bid1 + 1) {
                                    nb1 = 2;
                                }
                            }
                            const half *k1 = (const half *)args.key_cache
                                + bid1 * args.local_kv_head_num *
                                      args.block_size * args.size_per_head
                                + kv_head_idx * args.block_size *
                                      args.size_per_head;
                            const half *v1 = (const half *)args.value_cache
                                + bid1 * args.local_kv_head_num *
                                      args.block_size * args.size_per_head
                                + kv_head_idx * args.block_size *
                                      args.size_per_head;
                            memcpy_async(kseta[1], k1,
                                         BN * args.size_per_head *
                                             sizeof(half),
                                         MemcpyGlobalToSpm,
                                         kd_get_handle[1]);
                            if (nb1 == 2) {
                                memcpy_async(ksetb[1],
                                             k1 + args.local_kv_head_num *
                                                 args.block_size *
                                                 args.size_per_head,
                                             BN * args.size_per_head *
                                                 sizeof(half),
                                             MemcpyGlobalToSpm,
                                             kd2_get_handle[1]);
                            }
                            memcpy_async(vseta[1], v1,
                                         BN * args.size_per_head *
                                             sizeof(half),
                                         MemcpyGlobalToSpm,
                                         vd_get_handle[1]);
                            if (nb1 == 2) {
                                memcpy_async(vsetb[1],
                                             v1 + args.local_kv_head_num *
                                                 args.block_size *
                                                 args.size_per_head,
                                             BN * args.size_per_head *
                                                 sizeof(half),
                                             MemcpyGlobalToSpm,
                                             vd2_get_handle[1]);
                            }
                        }
                    }
                    while (j0 < last_k) {
                        int cur_block = j0 / args.block_size;
                        int block_id = ldm_block_table[cur_block];
                        int nb = 1;
                        if (j0 + 2 * BN <= last_k) {
                            int b1 = (j0 + BN) / args.block_size;
                            if (ldm_block_table[b1] == block_id + 1) {
                                nb = 2;
                            }
                        }
                        int span = nb * BN;
                        int cur_BN = MIN(span, last_k - j0);
                        int set = jd % 3;
                        half *cur_ka = kseta[set];
                        half *cur_kb = ksetb[set];
                        half *cur_va = vseta[set];
                        half *cur_vb = vsetb[set];
                        // issue the fetches for iteration jd+2 (two
                        // iterations ahead) into set (jd+2)%3.
                        int j2 = j0 + span;
                        if (j2 < last_k) {
                            int b1 = j2 / args.block_size;
                            int bid1 = ldm_block_table[b1];
                            int nb1 = 1;
                            if (j2 + 2 * BN <= last_k) {
                                int fb1 = (j2 + BN) / args.block_size;
                                if (ldm_block_table[fb1] == bid1 + 1) {
                                    nb1 = 2;
                                }
                            }
                            int j3 = j2 + nb1 * BN;
                            if (j3 < last_k) {
                                int nset = (jd + 2) % 3;
                                int b3 = j3 / args.block_size;
                                int bid3 = ldm_block_table[b3];
                                int nb3 = 1;
                                if (j3 + 2 * BN <= last_k) {
                                    int fb3 = (j3 + BN) / args.block_size;
                                    if (ldm_block_table[fb3] == bid3 + 1) {
                                        nb3 = 2;
                                    }
                                }
                                const half *nk_src =
                                    (const half *)args.key_cache
                                    + bid3 * args.local_kv_head_num *
                                          args.block_size *
                                          args.size_per_head
                                    + kv_head_idx * args.block_size *
                                          args.size_per_head;
                                const half *nv_src =
                                    (const half *)args.value_cache
                                    + bid3 * args.local_kv_head_num *
                                          args.block_size *
                                          args.size_per_head
                                    + kv_head_idx * args.block_size *
                                          args.size_per_head;
                                memcpy_async(kseta[nset], nk_src,
                                             BN * args.size_per_head *
                                                 sizeof(half),
                                             MemcpyGlobalToSpm,
                                             kd_get_handle[nset]);
                                if (nb3 == 2) {
                                    memcpy_async(ksetb[nset],
                                                 nk_src +
                                                     args.local_kv_head_num *
                                                     args.block_size *
                                                     args.size_per_head,
                                                 BN * args.size_per_head *
                                                     sizeof(half),
                                                 MemcpyGlobalToSpm,
                                                 kd2_get_handle[nset]);
                                }
                                memcpy_async(vseta[nset], nv_src,
                                             BN * args.size_per_head *
                                                 sizeof(half),
                                             MemcpyGlobalToSpm,
                                             vd_get_handle[nset]);
                                if (nb3 == 2) {
                                    memcpy_async(vsetb[nset],
                                                 nv_src +
                                                     args.local_kv_head_num *
                                                     args.block_size *
                                                     args.size_per_head,
                                                 BN * args.size_per_head *
                                                     sizeof(half),
                                                 MemcpyGlobalToSpm,
                                                 vd2_get_handle[nset]);
                                }
                            }
                        }
                        if (!(args.dbg_stub & 2048)) {
                        // FA-39 (DMA-wall probe, timing only): bits above
                        // skip the compute and keep only the async fetch
                        // pipeline. Outputs are garbage; measurement only.
                        // Q is constant across the K scan: convert the two
                        // group rows once per tile visit.
                        floatv16 qf0[4];
                        floatv16 qf1[4];
                        for (int h = 0; h < BK; h += 16) {
                            halfv16 qh;
                            simd_load(qh, qbuf + 0 * BK + h);
                            qf0[h / 16] = simd_vfcvths(qh);
                            simd_load(qh, qbuf + 1 * BK + h);
                            qf1[h / 16] = simd_vfcvths(qh);
                        }
                        for (int t = 0; t < cur_BN && t < BN; t++) {
                            floatv16 dot0 = 0.0f;
                            floatv16 dot1 = 0.0f;
                            for (int h = 0; h < BK; h += 16) {
                                halfv16 kh;
                                simd_load(kh, cur_ka + t * BK + h);
                                floatv16 kf = simd_vfcvths(kh);
                                dot0 = simd_vfmas(qf0[h / 16], kf, dot0);
                                dot1 = simd_vfmas(qf1[h / 16], kf, dot1);
                            }
                            outc[t] = simd_redsum(dot0) * args.softmax_scale;
                            outc[span + t] =
                                simd_redsum(dot1) * args.softmax_scale;
                        }
                        for (int t = BN; t < cur_BN; t++) {
                            floatv16 dot0 = 0.0f;
                            floatv16 dot1 = 0.0f;
                            for (int h = 0; h < BK; h += 16) {
                                halfv16 kh;
                                simd_load(kh, cur_kb + (t - BN) * BK + h);
                                floatv16 kf = simd_vfcvths(kh);
                                dot0 = simd_vfmas(qf0[h / 16], kf, dot0);
                                dot1 = simd_vfmas(qf1[h / 16], kf, dot1);
                            }
                            outc[t] = simd_redsum(dot0) * args.softmax_scale;
                            outc[span + t] =
                                simd_redsum(dot1) * args.softmax_scale;
                        }
                        // FA-42 diag: bit 8192 skips softmax+rescale
                        // (timing only, outputs garbage).
                        if (!(args.dbg_stub & 8192)) {
                        online_softmax_step(
                            outc, grp_m, grp_l, grp_mn, grp_ln, scale,
                            2, cur_BN, span);
                        for (int row = 0; row < 2; row++) {
                            // scale == 1.0 exactly when the running max is
                            // unchanged; skip the whole rescale pass then.
                            if (scale[row] == 1.0f) continue;
                            floatv16 row_scale = scale[row];
                            float *accum_row = o_accum + row * BK;
                            for (int h = 0; h < BK; h += 16) {
                                floatv16 accum_v;
                                simd_load(accum_v, accum_row + h);
                                simd_store(accum_v * row_scale, accum_row + h);
                            }
                        }
                        }
                        // FA-42 diag: bit 4096 skips the PV loop
                        // (timing only, outputs garbage).
                        if (!(args.dbg_stub & 4096)) {
                        for (int h = 0; h < BK; h += 16) {
                            floatv16 a0;
                            floatv16 a1;
                            simd_load(a0, o_accum + 0 * BK + h);
                            simd_load(a1, o_accum + 1 * BK + h);
                            for (int t = 0; t < cur_BN && t < BN; t++) {
                                halfv16 value_h;
                                simd_load(value_h, cur_va + t * BK + h);
                                floatv16 vf = simd_vfcvths(value_h);
                                float p0 = (float)(half)outc[t];
                                float p1 = (float)(half)outc[span + t];
                                a0 = simd_vfmas(vf, p0, a0);
                                a1 = simd_vfmas(vf, p1, a1);
                            }
                            for (int t = BN; t < cur_BN; t++) {
                                halfv16 value_h;
                                simd_load(value_h, cur_vb + (t - BN) * BK + h);
                                floatv16 vf = simd_vfcvths(value_h);
                                float p0 = (float)(half)outc[t];
                                float p1 = (float)(half)outc[span + t];
                                a0 = simd_vfmas(vf, p0, a0);
                                a1 = simd_vfmas(vf, p1, a1);
                            }
                            simd_store(a0, o_accum + 0 * BK + h);
                            simd_store(a1, o_accum + 1 * BK + h);
                        }
                        }
                        } else {
                            // consume one half per buffer so the fetches are
                            // not dead-code eliminated
                            fa39_probe += (float)cur_ka[0] + (float)cur_va[0];
                            if (nb == 2) {
                                fa39_probe += (float)cur_kb[0] +
                                              (float)cur_vb[0];
                            }
                        }
                        float *tmp = grp_l;
                        grp_l = grp_ln;
                        grp_ln = tmp;
                        tmp = grp_m;
                        grp_m = grp_mn;
                        grp_mn = tmp;
                        jd++;
                        j0 = j2;
                    }
                    if (grp_l != l_prev) {
                        for (int row = 0; row < 2; row++) {
                            l_prev[row] = grp_l[row];
                            m_prev[row] = grp_m[row];
                        }
                    }
                    if (args.dbg_stub & 2048) {
                        // observable sink for the probe accumulator
                        l_prev[0] += fa39_probe * 1e-30f;
                    }
    l_ptr = l_prev;
    m_ptr = m_prev;

    // Canonical phase-1 partial layout and byte order.
    const int partial_stride = BK + 2;
    for (int row = 0; row < 2; row++) {
        float *dst = (float *)args.workspace +
            (((head_idx + row) * 16 + split_idx) * partial_stride);
        outc[0] = m_ptr[row];
        outc[1] = l_ptr[row];
        memcpy(outc + 2, o_accum + row * BK, BK * sizeof(float));
        memcpy(dst, outc, partial_stride * sizeof(float));
    }

    // FA65: one arrival per split for this two-head group. The last arriver
    // executes the same split-order reducer as the canonical inline fold.
    int64_t *ctr = (int64_t *)args.fa65_ctr + head_task_idx;
    long long old_arr = sdaa::teco_atomic_add_64(ctr, 1);
    if ((old_arr + 1) % 16 != 0) return;

    float *partial_bulk =
        args.fa35
            ? (float *)rt_spm_malloc(8 * partial_stride * sizeof(float))
            : nullptr;
    float *accum = (float *)rt_spm_malloc(BK * sizeof(float));
    half *reduce_out = (half *)rt_spm_malloc(BK * sizeof(half));
    const float *ws_base = (const float *)args.workspace;
    for (int h_idx = head_idx; h_idx < head_idx + 2; h_idx++) {
        float running_m = -FLT_MAX;
        float running_l = 0.0f;
        for (int h = 0; h < BK; h += 16)
            simd_store(vzero, accum + h);
        const float *bulk_base =
            ws_base + (h_idx * 16) * partial_stride;
        for (int c0 = 0; c0 < 16; c0 += 8) {
            const int cn = MIN(8, 16 - c0);
            if (partial_bulk != nullptr) {
                memcpy(partial_bulk, bulk_base + c0 * partial_stride,
                       cn * partial_stride * sizeof(float));
            }
            for (int si = 0; si < cn; si++) {
                const float *src =
                    partial_bulk != nullptr
                        ? partial_bulk + si * partial_stride
                        : bulk_base + (c0 + si) * partial_stride;
                const float partial_l = src[1];
                if (partial_l <= 0.0f) continue;
                const float partial_m = src[0];
                const float next_m = MAX(running_m, partial_m);
                const float old_scale =
                    running_l > 0.0f ? expf(running_m - next_m) : 0.0f;
                const float partial_scale = expf(partial_m - next_m);
                for (int h = 0; h < BK; h++) {
                    accum[h] = accum[h] * old_scale +
                               src[2 + h] * partial_scale;
                }
                running_l = running_l * old_scale +
                            partial_l * partial_scale;
                running_m = next_m;
            }
        }
        const float l_inv =
            running_l > 0.0f ? 1.0f / running_l : 0.0f;
        const floatv16 vl_inv = l_inv;
        for (int h = 0; h < BK; h += 16) {
            floatv16 vf;
            halfv16 vh;
            simd_load(vf, accum + h);
            vh = (halfv16)(vf * vl_inv);
            simd_store(vh, reduce_out + h);
        }
        half *out = (half *)args.fmha_out + h_idx * BK;
        memcpy(out, reduce_out, BK * sizeof(half));
    }
}
