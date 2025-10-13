import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_idx_b, stride_idx_s, stride_idx_g,
    stride_kv_b, stride_kv_s, stride_kv_g,
    stride_out_b, stride_out_s, stride_out_h,
    stride_lse_b, stride_lse_s,
    sm_scale,
    PADDED_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    REPLICATE_H: tl.constexpr,
    H_PER_BLOCK: tl.constexpr,
    DIM: tl.constexpr,
    TAIL_DIM: tl.constexpr,
    TOP_K: tl.constexpr,
):
    bx = tl.program_id(0)
    by = tl.program_id(1)
    bz = tl.program_id(2)
    b_i, g_i = by, bz
    if REPLICATE_H == 1 and bx % 2 == 0:
        return
    s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
    q_i = s_i
    max_kv_i = q_i
    H0 = g_i * PADDED_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
    h_offs = tl.arange(0, H_PER_BLOCK)
    dim_offs = tl.arange(0, DIM)
    dim_tail_offs = DIM + tl.arange(0, TAIL_DIM)

    q_offs_base = b_i * stride_qb + s_i * stride_qs + (H0 + h_offs[:, None])*stride_qh
    q = tl.load(q_ptr + q_offs_base + dim_offs[None,:])
    q_tail = tl.load(q_ptr + q_offs_base + dim_tail_offs[None,:])

    block_i_offs = tl.arange(0, BLOCK_I)
    m_i = tl.full(shape=[H_PER_BLOCK], value=-(2**10), dtype=tl.float32)
    sum_exp = tl.zeros(shape=[H_PER_BLOCK], dtype=tl.float32)
    acc_o = tl.zeros(shape=[H_PER_BLOCK, DIM], dtype=tl.float32)
    for i_i in tl.range(0, tl.cdiv(TOP_K, BLOCK_I)):
        indices_offs_base = b_i * stride_idx_b + s_i * stride_idx_s + g_i * stride_idx_g
        indices = tl.load(indices_ptr + indices_offs_base + (i_i * BLOCK_I + block_i_offs))
        mask_idx = (indices <= max_kv_i)
        kv_offs_base = b_i * stride_kv_b + indices * stride_kv_s + g_i * stride_kv_g
        kv_mask = indices[:,None] <= max_kv_i
        kv = tl.load(kv_ptr + kv_offs_base[:,None] + dim_offs[None,:], mask=kv_mask, other=0.0)
        kv_tail = tl.load(kv_ptr + kv_offs_base[:,None] + dim_tail_offs[None,:], mask=kv_mask, other=0.0)
        
        mask_idx_acc = tl.broadcast_to(mask_idx.reshape(1, BLOCK_I), (H_PER_BLOCK, BLOCK_I))
        acc_s = tl.zeros(shape=[H_PER_BLOCK, BLOCK_I], dtype=tl.float32)
        acc_s = tl.where(mask_idx_acc, acc_s, -(2**10)) #)-float('inf')) # origin is -inf
        acc_s = tl.dot(q, tl.trans(kv), acc=acc_s)
        acc_s = tl.dot(q_tail, tl.trans(kv_tail), acc=acc_s)
        m_i_prev = m_i
        m_i = tl.max(acc_s, axis=1)
        
        alpha = tl.math.exp2((m_i_prev - m_i) * sm_scale)
        m_i_brd = tl.broadcast_to(m_i.reshape(H_PER_BLOCK, 1), (H_PER_BLOCK, BLOCK_I))
        acc_s = tl.math.exp2(acc_s * sm_scale - m_i_brd * sm_scale)
        sum_exp_i = tl.sum(acc_s, axis=1)
        sum_exp = sum_exp * alpha + sum_exp_i
        acc_o = acc_o * tl.broadcast_to(alpha.reshape(H_PER_BLOCK,1), (H_PER_BLOCK, DIM))
        acc_o = tl.dot(acc_s.to(kv_ptr.dtype.element_ty), kv, acc=acc_o)
    acc_o = acc_o / sum_exp[:, None]
    sum_exp = tl.math.log2(sum_exp) + m_i * sm_scale
    out_offs_base = b_i * stride_out_b + s_i * stride_out_s + (H0 + h_offs[:, None])*stride_out_h
    tl.store(out_ptr + out_offs_base + dim_offs[None,:], acc_o.to(out_ptr.dtype.element_ty))
    tl.store(lse_ptr + b_i * stride_lse_b + s_i * stride_lse_s + (H0 + h_offs), sum_exp)

def sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    # kernel = sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual)
    # out, lse = kernel(q, kv, indices)
    assert dim == triton.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == triton.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"

    head_kv = heads // kv_group
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    BLOCK_I = 64
    assert (topk %
            BLOCK_I == 0), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)
    padded_H = max(triton.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert (
            kv_group == 1
        ), "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    out = torch.empty((batch, seq_len, heads, dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, seq_len, heads), dtype=torch.float32, device=q.device)
    grid = (seq_len * REPLICATE_H, batch, kv_group)
    _kernel[grid](
        q_ptr = q,
        kv_ptr = kv,
        indices_ptr = indices,
        out_ptr = out,
        lse_ptr = lse,
        stride_qb = q.stride(0), stride_qs = q.stride(1), stride_qh = q.stride(2),
        stride_idx_b = indices.stride(0), stride_idx_s = indices.stride(1), stride_idx_g = indices.stride(2),
        stride_kv_b = kv.stride(0), stride_kv_s = kv.stride(1), stride_kv_g = kv.stride(2),
        stride_out_b = out.stride(0), stride_out_s = out.stride(1), stride_out_h = out.stride(2),
        stride_lse_b = lse.stride(0), stride_lse_s = lse.stride(1),
        sm_scale = sm_scale,
        PADDED_H = padded_H,
        BLOCK_I = BLOCK_I,
        REPLICATE_H = REPLICATE_H,
        H_PER_BLOCK = H_per_block,
        DIM = dim,
        TAIL_DIM = tail_dim,
        TOP_K = topk,
    )
    
    return out, lse
