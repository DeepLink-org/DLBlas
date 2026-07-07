# ============================================================================
# MTPBlock Golden 计算（所有 kernel 的参考实现）
# ============================================================================

import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    # clip to avoid overflow
    x = np.clip(x, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_golden_k2_hc_pre(x, hc_fn, hc_scale, hc_base, hc_sinkhorn_iters=20, eps=1e-6):
    """
    K2 hc_pre 参考实现 (matching PyTorch reference)

    Inputs:
        x:         [b, s, hc, d]   bf16 values as float32
        hc_fn:     [mix_hc, hc*d]  float32
        hc_scale:  [3]             float32
        hc_base:   [mix_hc]        float32

    Returns:
        y:    [b, s, d]          float32
        pre:  [b, s, hc]         float32
        post: [b, s, hc]         float32
        comb: [b, s, hc, hc]     float32
    """
    b, s, hc, d = x.shape
    hcd = hc * d
    mix_hc = hc_fn.shape[0]
    dim_bs = b * s

    # Flatten
    x_flat = x.reshape(dim_bs, hcd).astype(np.float32)

    # RMSNorm
    mean_sq = np.mean(x_flat ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(mean_sq + eps)

    # Linear projection
    mixes = (x_flat @ hc_fn.T) * rsqrt  # [dim_bs, mix_hc]

    # Scale + bias
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]
    x_scaled = mixes.copy()
    x_scaled[:, :hc] = x_scaled[:, :hc] * s0 + hc_base[:hc]
    x_scaled[:, hc:2*hc] = x_scaled[:, hc:2*hc] * s1 + hc_base[hc:2*hc]
    x_scaled[:, 2*hc:] = x_scaled[:, 2*hc:] * s2 + hc_base[2*hc:]

    # Split + sigmoid
    pre = sigmoid(x_scaled[:, :hc]) + eps
    post = 2.0 * sigmoid(x_scaled[:, hc:2*hc])
    comb = x_scaled[:, 2*hc:].reshape(dim_bs, hc, hc)

    # Sinkhorn
    comb = comb.copy()
    # softmax along dim=-1
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / (comb.sum(axis=-1, keepdims=True) + eps) + eps
    # col normalize
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    # row-norm → col-norm × (iters-1)
    for _ in range(hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    # Weighted sum
    x_reshaped = x.astype(np.float32)
    y = np.sum(pre.reshape(b, s, hc, 1) * x_reshaped, axis=2)  # [b, s, d]

    return y, pre.reshape(b, s, hc), post.reshape(b, s, hc), comb.reshape(b, s, hc, hc)


def compute_golden_k1_embed_fuse(x, input_ids, embed_weight, enorm_weight,
                                  e_proj_weight, h_proj_weight, hnorm_weight, eps=1e-6):
    """
    K1 embed_fuse 参考实现

    Args:
        x:              [b, s, hc, d]   bf16 values as float32
        input_ids:      [b, s]          int64
        embed_weight:   [vocab, d]      bf16 values as float32
        enorm_weight:   [d]             float32
        e_proj_weight:  [d, d]          bf16 values as float32
        h_proj_weight:  [d, d]          bf16 values as float32
        hnorm_weight:   [d]             float32

    Returns:
        feat: [b, s, hc, d] float32
    """
    b, s, hc, d = x.shape

    # Embedding lookup
    e = embed_weight[input_ids]  # [b, s, d]

    # enorm RMSNorm: e = e * rsqrt(mean(e²)+eps) * enorm_weight
    e_f32 = e.astype(np.float32)
    enorm_w = enorm_weight.astype(np.float32)
    mean_sq_e = np.mean(e_f32 ** 2, axis=-1, keepdims=True)
    rsqrt_e = 1.0 / np.sqrt(mean_sq_e + eps)
    e_normed = e_f32 * rsqrt_e * enorm_w  # [b, s, d]

    # e_proj: [b*s, d] × [d, d]^T → [b*s, d]
    e_flat = e_normed.reshape(b * s, d)
    e_proj = e_flat @ e_proj_weight.astype(np.float32).T  # [b*s, d]

    # hnorm RMSNorm on x: per [hc, d] plane
    x_f32 = x.astype(np.float32)
    hnorm_w = hnorm_weight.astype(np.float32)
    mean_sq_x = np.mean(x_f32 ** 2, axis=-1, keepdims=True)
    rsqrt_x = 1.0 / np.sqrt(mean_sq_x + eps)
    x_normed = x_f32 * rsqrt_x * hnorm_w  # [b, s, hc, d]

    # h_proj: [b*s*hc, d] × [d, d]^T → [b*s*hc, d]
    x_flat = x_normed.reshape(b * s * hc, d)
    h_proj = x_flat @ h_proj_weight.astype(np.float32).T  # [b*s*hc, d]
    h_proj = h_proj.reshape(b, s, hc, d)

    # Broadcast add: feat = unsqueeze(e_proj, 2) + h_proj
    feat = e_proj.reshape(b, s, 1, d) + h_proj  # [b, s, hc, d]

    return feat


def compute_golden_k4_hc_post(x, residual, post, comb):
    """
    K4 hc_post 参考实现 (matching PyTorch reference exactly)

    PyTorch reference:
      post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    Equivalent matrix form (per (b,s)):
      out[j,:] = post[j] * x[:] + sum_i comb[i,j] * residual[j,:]

    Args:
        x:        [b, s, d]       float32
        residual: [b, s, hc, d]   float32
        post:     [b, s, hc]      float32
        comb:     [b, s, hc, hc]  float32

    Returns:
        out: [b, s, hc, d] float32
    """
    b, s, hc, d = x.shape[0], x.shape[1], residual.shape[2], x.shape[2]
    out = np.zeros((b, s, hc, d), dtype=np.float32)

    for b_idx in range(b):
        for si in range(s):
            for hc_i in range(hc):  # output index = j
                # post_term: post[j] * x[:]
                out[b_idx, si, hc_i, :] = post[b_idx, si, hc_i] * x[b_idx, si, :]
                # comb_term: sum_i comb[i,j] * residual[j,:]
                for hc_j in range(hc):  # sum index = i
                    out[b_idx, si, hc_i, :] += (
                        comb[b_idx, si, hc_j, hc_i] * residual[b_idx, si, hc_i, :]
                    )
    return out


def compute_golden_k3_attn_block(x, wq_a_weight, q_norm_weight, wq_b_weight,
                                  wkv_weight, kv_norm_weight, wo_a_weight,
                                  wo_b_weight, attn_sink, softmax_scale, eps=1e-6):
    """
    K3 attn_block 参考实现

    Q branch: wq_a → RMSNorm → wq_b → reshape → RMSNorm
    KV branch: wkv → RMSNorm
    Attention: QK^T → softmax with sink → weighted sum
    Output: wo_a (grouped) → wo_b

    Args:
        x:               [s, d]            bf16 as float32
        wq_a_weight:     [q_lora, d]       bf16 as float32
        q_norm_weight:   [q_lora]          float32
        wq_b_weight:     [n_heads*head_dim, q_lora]  bf16 as float32
        wkv_weight:      [head_dim, d]     bf16 as float32
        kv_norm_weight:  [head_dim]        float32
        wo_a_weight:     [n_groups, o_lora, n_heads*head_dim/n_groups]  bf16 as float32
        wo_b_weight:     [d, n_groups*o_lora]  bf16 as float32
        attn_sink:       [n_heads]         float32
        softmax_scale:   float
        eps:             float

    Returns:
        attn_out: [s, d] float32
    """
    s = x.shape[0]
    d = x.shape[1]
    q_lora = wq_a_weight.shape[0]
    n_heads = attn_sink.shape[0]
    head_dim = wkv_weight.shape[0]
    o_lora = wo_a_weight.shape[1]
    n_groups = wo_a_weight.shape[0]
    hpg = n_heads * head_dim // n_groups  # heads per group
    nhd = n_heads * head_dim

    x_f32 = x.astype(np.float32)
    wq_a = wq_a_weight.astype(np.float32)
    q_norm = q_norm_weight.astype(np.float32)
    wq_b = wq_b_weight.astype(np.float32)
    wkv = wkv_weight.astype(np.float32)
    kv_norm = kv_norm_weight.astype(np.float32)
    wo_a = wo_a_weight.astype(np.float32)
    wo_b = wo_b_weight.astype(np.float32)

    # Q Branch: q1 = RMSNorm(x @ wq_a^T)
    q1 = x_f32 @ wq_a.T  # [s, q_lora]
    mean_sq = np.mean(q1 ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(mean_sq + eps)
    q1 = q1 * rsqrt * q_norm  # [s, q_lora]

    # q2 = q1 @ wq_b^T → reshape to [s, n_heads, head_dim]
    q2 = q1 @ wq_b.T  # [s, nhd]
    q = q2.reshape(s, n_heads, head_dim)

    # Q RMSNorm per head
    mean_sq_h = np.mean(q ** 2, axis=-1, keepdims=True)
    rsqrt_h = 1.0 / np.sqrt(mean_sq_h + eps)
    q = q * rsqrt_h  # [s, n_heads, head_dim]

    # KV Branch: kv = RMSNorm(x @ wkv^T)
    kv = x_f32 @ wkv.T  # [s, head_dim]
    mean_sq_kv = np.mean(kv ** 2, axis=-1, keepdims=True)
    rsqrt_kv = 1.0 / np.sqrt(mean_sq_kv + eps)
    kv = kv * rsqrt_kv * kv_norm  # [s, head_dim]

    # Attention: scores[si, hi, sj] = dot(q[si,hi], kv[sj]) * scale
    scores = np.zeros((s, n_heads, s), dtype=np.float32)
    for si in range(s):
        for hi in range(n_heads):
            for sj in range(s):
                scores[si, hi, sj] = np.dot(q[si, hi, :], kv[sj, :]) * softmax_scale

    # Softmax with attn_sink in denominator only
    attn_out = np.zeros((s, n_heads, head_dim), dtype=np.float32)
    for si in range(s):
        for hi in range(n_heads):
            row = scores[si, hi, :]
            mx = max(np.max(row), attn_sink[hi])
            exp_row = np.exp(row - mx)
            exp_sink = np.exp(attn_sink[hi] - mx)
            se = np.sum(exp_row) + exp_sink
            weights = exp_row / se

            # Weighted sum
            for di in range(head_dim):
                attn_out[si, hi, di] = np.dot(weights, kv[:, di])

    # Output projection: wo_a (grouped)
    # Flatten attn_out [s, n_heads, head_dim] → [s, nhd]
    ao_flat = attn_out.reshape(s, nhd).astype(np.float32)
    # o1[si, gi, n] = sum_k ao_flat[si, gi*hpg + k] * wo_a[gi, n, k]
    o1 = np.zeros((s, n_groups, o_lora), dtype=np.float32)
    for si in range(s):
        for gi in range(n_groups):
            base_k = gi * hpg
            for n in range(o_lora):
                dot = 0.0
                for k in range(hpg):
                    dot += ao_flat[si, base_k + k] * wo_a[gi, n, k]
                o1[si, gi, n] = dot

    # wo_b: o2[si, :] = o1_flat @ wo_b^T
    o1_flat = o1.reshape(s, n_groups * o_lora)
    o2 = o1_flat @ wo_b.T  # [s, d]

    return o2


def compute_golden_k5_moe_block(x, gate_weight, gate_bias,
                                  shared_w1, shared_w2, shared_w3, eps=1e-6):
    """
    K5 moe_block 参考实现 (Shared Expert only for demo)

    gate = SiLU(w1 @ x) * (w3 @ x)
    out  = w2 @ gate

    Args:
        x:            [b*s, d]       bf16 as float32
        gate_weight:  [n_experts, d] bf16 as float32 (unused in demo)
        gate_bias:    [n_experts]    float32 (unused in demo)
        shared_w1:    [inter_dim, d] bf16 as float32
        shared_w2:    [d, inter_dim] bf16 as float32
        shared_w3:    [inter_dim, d] bf16 as float32
        eps:          float (unused)

    Returns:
        output: [b*s, d] float32
    """
    bs = x.shape[0]
    d = x.shape[1]
    inter_dim = shared_w1.shape[0]

    x_f32 = x.astype(np.float32)
    sw1 = shared_w1.astype(np.float32)
    sw2 = shared_w2.astype(np.float32)
    sw3 = shared_w3.astype(np.float32)

    # gate = SiLU(w1 @ x^T)
    gate = x_f32 @ sw1.T  # [bs, inter_dim]
    # SiLU: gate = x * sigmoid(x)
    gate = gate * sigmoid(gate)  # [bs, inter_dim]

    # up = w3 @ x^T
    up = x_f32 @ sw3.T  # [bs, inter_dim]

    # combined = gate * up
    combined = gate * up  # [bs, inter_dim]

    # out = combined @ w2^T
    out = combined @ sw2.T  # [bs, d]

    return out


def compute_golden_k6_mtp_head(x, hc_head_fn, hc_head_scale, hc_head_base,
                                norm_weight, head_weight, eps=1e-6):
    """
    K6 mtp_head 参考实现

    1. Flatten x [b,s,hc,d] → [b*s, hc*d]
    2. RMSNorm
    3. mixes = x_flat @ hc_fn^T * rsqrt [b*s, hc]
    4. pre = sigmoid(mixes * scale + base) + eps
    5. y = sum(pre * x, dim=2) [b,s,d]
    6. x_last = RMSNorm(y[:,-1]) [b,d]
    7. logits = x_last @ head_weight^T [b, vocab]

    Args:
        x:              [b, s, hc, d]  bf16 as float32
        hc_head_fn:     [hc, hc*d]     float32
        hc_head_scale:  [1]            float32
        hc_head_base:   [hc]           float32
        norm_weight:    [d]            float32
        head_weight:    [vocab, d]     float32
        eps:            float

    Returns:
        logits: [b, vocab] float32
    """
    b, s, hc, d = x.shape
    hcd = hc * d
    dim_bs = b * s
    vocab = head_weight.shape[0]

    x_f32 = x.astype(np.float32)
    head_w = head_weight.astype(np.float32)
    norm_w = norm_weight.astype(np.float32)

    # Flatten
    x_flat = x_f32.reshape(dim_bs, hcd)

    # RMSNorm
    mean_sq = np.mean(x_flat ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(mean_sq + eps)

    # Linear projection
    mixes = (x_flat @ hc_head_fn.T) * rsqrt  # [dim_bs, hc]

    # pre = sigmoid(mixes * scale + base) + eps
    hc_scale_val = hc_head_scale[0]
    pre = sigmoid(mixes * hc_scale_val + hc_head_base) + eps  # [dim_bs, hc]

    # Weighted sum y[m, di] = sum_i pre[m,i] * x[m, i*d+di]
    y = np.zeros((dim_bs, d), dtype=np.float32)
    for m in range(dim_bs):
        for di in range(d):
            acc = 0.0
            for i in range(hc):
                acc += pre[m, i] * x_f32.reshape(dim_bs, hc, d)[m, i, di]
            y[m, di] = acc
    y = y.reshape(b, s, d)

    # RMSNorm on last token y[:,-1,:]
    x_last = y[:, -1, :]  # [b, d]
    mean_sq_last = np.mean(x_last ** 2, axis=-1, keepdims=True)
    rsqrt_last = 1.0 / np.sqrt(mean_sq_last + eps)
    x_last_normed = x_last * rsqrt_last * norm_w  # [b, d]

    # logits = x_last_normed @ head_weight^T
    logits = x_last_normed @ head_w.T  # [b, vocab]

    return logits
