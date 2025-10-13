import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    Q,
    K,
    KScale,
    Logits,
    Weights,
    CuSeqLenKS,
    CuSeqLenKE,
    seq_len_kv,
    heads: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bx = tl.program_id(0)
    seq_len_i = bx * BLOCK_Q
    seq_len_offs = seq_len_i + tl.arange(0, BLOCK_Q)
    
    cu_seqlen_ks = tl.load(CuSeqLenKS + seq_len_offs)
    cu_k_s_min = tl.minimum(tl.min(cu_seqlen_ks), seq_len_kv)
    cu_seqlen_ke = tl.load(CuSeqLenKE + seq_len_offs)
    cu_k_e_max = tl.minimum(tl.max(cu_seqlen_ke), seq_len_kv) # ke max, skv, skv
    dim_offs = tl.arange(0, dim)[None, :]

    q = tl.load(Q + seq_len_i * heads + tl.arange(0, BLOCK_Q * heads)[:, None] + dim_offs)
    q_t = tl.trans(q)
    heads_offs = tl.arange(0, heads)[None, :]
    w = tl.load(Weights + seq_len_offs[:, None] + heads_offs)
    w = tl.broadcast_to(w.reshape(1, BLOCK_Q * heads), (BLOCK_N, BLOCK_Q * heads))
    block_n_offs = tl.arange(0, BLOCK_N)
    for nbn_i in tl.range(0, tl.cdiv(cu_k_e_max - cu_k_s_min, BLOCK_N)):
    # k [BN, dim]   q[bq * heads, dim] = s [BN, bq * heads]
        k = tl.load(K + cu_k_s_min + nbn_i * BLOCK_N + block_n_offs[:,None] + dim_offs)
        k_scale = tl.load(KScale + cu_k_s_min + nbn_i * BLOCK_N + block_n_offs)

        accumulator = tl.dot(k, q_t)
        s = tl.where(accumulator > 0, accumulator, 0)
        s_reshaped = s * w * tl.broadcast_to(k_scale.reshape(BLOCK_N, 1), (BLOCK_N, BLOCK_Q * heads))
        out = tl.sum(s_reshaped.reshape(BLOCK_N, BLOCK_Q, heads), axis=2)
        out_offs = seq_len_offs[:, None] + cu_k_s_min + nbn_i * BLOCK_N + block_n_offs[None, :]
        tl.store(Logits + out_offs, tl.trans(out))


def lighting_indexer(q, kv, kv_scales, logits, weights, cu_seqlen_ks, cu_seqlen_ke):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    BLOCK_Q = 128 // heads
    BLOCK_N = 256
    grid = (triton.cdiv(seq_len, BLOCK_Q), )
    _kernel[grid](
        Q=q.view(seq_len * heads, index_dim),
        K = kv,
        KScale=kv_scales,
        Logits=logits,
        Weights=weights,
        CuSeqLenKS=cu_seqlen_ks,
        CuSeqLenKE=cu_seqlen_ke,
        seq_len_kv=seq_len_kv,
        heads=heads,
        dim = index_dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_N=BLOCK_N,
    )
    return logits


def mqa_attn_return_logits_interface(q,
                                     kv,
                                     kv_scales,
                                     weights,
                                     cu_seqlen_ks,
                                     cu_seqlen_ke,
                                     clean_logits=True,
                                     use_triton=True):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    lighting_indexer(
        q,
        kv,
        kv_scales,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return logits
