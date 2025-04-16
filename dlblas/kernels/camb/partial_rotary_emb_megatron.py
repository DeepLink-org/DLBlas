# Copyright (c) 2025, DeepLink.
import torch
import triton
import triton.language as tl

from dlblas.utils import ChoiceSpace, SymVar, Tensor, register_dlblas_op


@triton.jit
def _new_get_cos_sin(
    batch_idx,
    offs_seq,
    rope_dim_range,
    rope_dim,
    end,
    seq_len,
    rope_head_dim,
    cos,
    sin,
    stride_cos_bsz,
    stride_cos_seq,
    stride_cos_dim,
):
    offs_cs = (batch_idx * stride_cos_bsz + offs_seq[:, None] * stride_cos_seq +
               rope_dim_range[None, :] * stride_cos_dim)
    cos_data = tl.load(
        cos + offs_cs,
        mask=(offs_seq[:, None] < seq_len),
    )
    sin_data = tl.load(
        sin + offs_cs,
        mask=(offs_seq[:, None] < seq_len),
    )
    cos0_data = cos_data[:, 0:rope_dim]
    cos1_data = cos_data[:, rope_dim:end]
    sin0_data = sin_data[:, 0:rope_dim]
    sin1_data = sin_data[:, rope_dim:end]
    return cos0_data, cos1_data, sin0_data, sin1_data


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SEQ': BS,
            'BLOCK_HEAD': BH
        }, num_stages=s, num_warps=w) for BS in [1024] for BH in [4] for s in [4] for w in [4]
    ],
    key=['seq_len', 'q_head_dim'],
)
@triton.jit
def _partial_rotary_emb_fwd_kernel(
    q,
    k_pe,
    k_nope,
    cos,
    sin,
    out_k,
    stride_q_bsz,
    stride_q_seq,
    stride_q_head,
    stride_q_dim,
    stride_kpe_bsz,
    stride_kpe_seq,
    stride_kpe_head,
    stride_kpe_dim,
    stride_cos_bsz,
    stride_cos_seq,
    stride_cos_dim,
    stride_okv_bsz,
    stride_okv_seq,
    stride_okv_head,
    stride_okv_dim,
    stride_kv_bsz,
    stride_kv_seq,
    stride_kv_head,
    stride_kv_dim,
    seq_len,
    q_head_dim,
    rope_head_dim,
    nope_head_dim,
    num_heads,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_ROPE_DIM: tl.constexpr,
    BLOCK_NOPE_DIM: tl.constexpr,
):
    stride_okv_kv = 1
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    batch_head_idx = tl.program_id(2)
    head_start = batch_head_idx * BLOCK_HEAD
    head_end = (batch_head_idx + 1) * BLOCK_HEAD
    offs_seq = seq_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    nope_dims = tl.arange(0, BLOCK_NOPE_DIM)
    rope_dim_range0 = tl.arange(0, BLOCK_ROPE_DIM // 2)
    rope_dim_range1 = tl.arange(BLOCK_ROPE_DIM // 2, BLOCK_ROPE_DIM)
    rope_dim_range = tl.arange(0, BLOCK_ROPE_DIM)
    rope_dim0_interleaved = 2 * rope_dim_range0
    rope_dim1_interleaved = 1 + 2 * rope_dim_range0
    rope_dim_interleaved = tl.arange(0, BLOCK_ROPE_DIM)

    #load cos & sin
    offs_cs = (batch_idx * stride_cos_bsz + offs_seq[:, None] * stride_cos_seq +
               rope_dim_range[None, :] * stride_cos_dim)

    cos_data = tl.load(
        cos + offs_cs,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range[None, :] < rope_head_dim),
    )

    sin_data = tl.load(
        sin + offs_cs,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range[None, :] < rope_head_dim),
    )

    #load kpe
    offs_kpe = (batch_idx * stride_kpe_bsz + offs_seq[:, None] * stride_kpe_seq +
                rope_dim_interleaved[None, :] * stride_kpe_dim)

    kpe_data = tl.load(
        k_pe + offs_kpe,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim_interleaved[None, :] < rope_head_dim),
    )

    #compute kpe
    out_kpe0_data = kpe_data[:, 0:BLOCK_ROPE_DIM:2] * cos_data[:, :BLOCK_ROPE_DIM //
                                                               2] - kpe_data[:, 1:BLOCK_ROPE_DIM:
                                                                             2] * sin_data[:, :BLOCK_ROPE_DIM // 2]
    out_kpe1_data = kpe_data[:, 1:BLOCK_ROPE_DIM:2] * cos_data[:, BLOCK_ROPE_DIM //
                                                               2:] + kpe_data[:, 0:BLOCK_ROPE_DIM:
                                                                              2] * sin_data[:, BLOCK_ROPE_DIM // 2:]

    for head_idx in tl.range(head_start, head_end):
        offs_q = (batch_idx * stride_q_bsz + offs_seq[:, None] * stride_q_seq + head_idx * stride_q_head)
        q_data = tl.load(
            q + offs_q + (nope_head_dim + rope_dim_interleaved[None, :]) * stride_q_dim,
            mask=(offs_seq[:, None] < seq_len)
            & (rope_dim_interleaved[None, :] < q_head_dim),
        )

        #compute q
        out_q0_data = q_data[:, 0:BLOCK_ROPE_DIM:2] * cos_data[:, :BLOCK_ROPE_DIM //
                                                               2] - q_data[:, 1:BLOCK_ROPE_DIM:
                                                                           2] * sin_data[:, :BLOCK_ROPE_DIM // 2]
        out_q1_data = q_data[:, 1:BLOCK_ROPE_DIM:2] * cos_data[:, BLOCK_ROPE_DIM //
                                                               2:] + q_data[:, 0:BLOCK_ROPE_DIM:
                                                                            2] * sin_data[:, BLOCK_ROPE_DIM // 2:]
        offs_q0 = (batch_idx * stride_q_bsz + offs_seq[:, None] * stride_q_seq + head_idx * stride_q_head)
        offs_q1 = (batch_idx * stride_q_bsz + offs_seq[:, None] * stride_q_seq + head_idx * stride_q_head)
        # store q
        tl.store(
            q + offs_q0 + (nope_head_dim + rope_dim_range0[None, :]) * stride_q_dim,
            out_q0_data,
            mask=(offs_seq[:, None] < seq_len)
            & (rope_dim_range0[None, :] < q_head_dim),
        )

        tl.store(
            q + offs_q1 + (nope_head_dim + rope_dim_range1[None, :]) * stride_q_dim,
            out_q1_data,
            mask=(offs_seq[:, None] < seq_len)
            & (rope_dim_range1[None, :] < q_head_dim),
        )
        #load k
        offs_kv = (batch_idx * stride_kv_bsz + offs_seq[:, None] * stride_kv_seq + head_idx * stride_kv_head)
        k_nope_data = tl.load(
            k_nope + offs_kv + nope_dims * stride_kv_dim,
            mask=(offs_seq[:, None] < seq_len) & (nope_dims[None, :] < nope_head_dim),
        )

        # store k
        offs_okv_k_nope = (batch_idx * stride_okv_bsz + offs_seq[:, None] * stride_okv_seq +
                           head_idx * stride_okv_head + nope_dims * stride_okv_dim)
        tl.store(
            out_k + offs_okv_k_nope,
            k_nope_data,
            mask=(offs_seq[:, None] < seq_len) & (nope_dims[None, :] < nope_head_dim),
        )

        offs_okv_kpe = (batch_idx * stride_okv_bsz + offs_seq[:, None] * stride_okv_seq + head_idx * stride_okv_head)

        tl.store(
            out_k + offs_okv_kpe + (nope_head_dim + rope_dim_range0) * stride_okv_dim,
            out_kpe0_data,
            mask=(offs_seq[:, None] < seq_len)
            & (rope_dim_range0[None, :] < rope_head_dim),
        )
        tl.store(
            out_k + offs_okv_kpe + (nope_head_dim + rope_dim_range1) * stride_okv_dim,
            out_kpe1_data,
            mask=(offs_seq[:, None] < seq_len)
            & (rope_dim_range1[None, :] < rope_head_dim),
        )


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SEQ': BS,
            'BLOCK_HEAD': BH
        }, num_stages=s, num_warps=w) for BS in [1024] for BH in [4] for s in [4] for w in [4]
    ],
    key=['seq_len', 'q_head_dim', 'rope_head_dim'],
)
@triton.jit
def _partial_rotary_emb_bwd_kernel(
    d_q_out,
    d_k_out,
    cos,
    sin,
    d_k_pe_in,
    d_k_nope_in,
    stride_dq_bsz,
    stride_dq_seq,
    stride_dq_head,
    stride_dq_dim,
    stride_dk_out_bsz,
    stride_dk_out_seq,
    stride_dk_out_head,
    stride_dk_out_dim,
    stride_cos_bsz,
    stride_cos_seq,
    stride_cos_dim,
    stride_dokpe_bsz,
    stride_dokpe_seq,
    stride_dokpe_head,
    stride_dokpe_dim,
    stride_k_nope_in_bsz,
    stride_k_nope_in_seq,
    stride_k_nope_in_head,
    stride_k_nope_in_dim,
    seq_len,
    q_head_dim,
    rope_head_dim,
    nope_head_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_ROPE_DIM: tl.constexpr,
    BLOCK_NOPE_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    batch_head_idx = tl.program_id(2)
    head_start = batch_head_idx * BLOCK_HEAD
    head_end = (batch_head_idx + 1) * BLOCK_HEAD
    offs_seq = seq_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    nope_dim_range = tl.arange(0, BLOCK_NOPE_DIM)
    rope_dim_range = tl.arange(0, BLOCK_ROPE_DIM)
    rope_dim_range0 = tl.arange(0, BLOCK_ROPE_DIM // 2)
    rope_dim_range1 = tl.arange(BLOCK_ROPE_DIM // 2, BLOCK_ROPE_DIM)
    rope_dim0_interleaved = 2 * rope_dim_range0
    rope_dim1_interleaved = 1 + 2 * rope_dim_range0
    rope_dim_interleaved = rope_dim_range

    #load cos & sin
    cos0_data, cos1_data, sin0_data, sin1_data = _new_get_cos_sin(
        batch_idx,
        offs_seq,
        rope_dim_range,
        BLOCK_ROPE_DIM // 2,
        BLOCK_ROPE_DIM,
        seq_len,
        rope_head_dim,
        cos,
        sin,
        stride_cos_bsz,
        stride_cos_seq,
        stride_cos_dim,
    )

    # for do_k_pe
    offs_d_kv = (batch_idx * stride_dk_out_bsz + offs_seq[:, None] * stride_dk_out_seq)

    d_k_pe_data = tl.load(
        d_k_out + offs_d_kv + (nope_head_dim + rope_dim_range[None, :]) * stride_dk_out_dim,
        mask=(offs_seq[:, None] < seq_len) & (rope_dim_range[None, :] < q_head_dim),
    )

    d_k_pe0_data = d_k_pe_data[:, :BLOCK_ROPE_DIM // 2]
    d_k_pe1_data = d_k_pe_data[:, BLOCK_ROPE_DIM // 2:]

    do_kpe_data = tl.empty_like(d_k_pe_data)
    do_kpe_data[:, :BLOCK_ROPE_DIM // 2] = (d_k_pe1_data * sin1_data + d_k_pe0_data * cos0_data).to(tl.bfloat16)
    do_kpe_data[:, BLOCK_ROPE_DIM // 2:] = (d_k_pe1_data * cos1_data - d_k_pe0_data * sin0_data).to(tl.bfloat16)

    do_kpe_data_view = tl.view(do_kpe_data, [BLOCK_SEQ, BLOCK_ROPE_DIM // 2, 2])
    do_kpe_data_view = tl.trans(do_kpe_data_view, 2, 0, 1)

    offs_d_kpe_in = (batch_idx * stride_dokpe_bsz + offs_seq[:, None] * stride_dokpe_seq)
    tl.store(
        d_k_pe_in + offs_d_kpe_in + rope_dim_range[None, :],
        do_kpe_data,
        mask=(offs_seq[:, None] < seq_len)
        & (rope_dim_range[None, :] < rope_head_dim),
    )

    for head_idx in tl.range(head_start, head_end):
        #load q
        offs_dq = (batch_idx * stride_dq_bsz + offs_seq[:, None] * stride_dq_seq + head_idx * stride_dq_head)
        dq_data = tl.load(
            d_q_out + offs_dq + (nope_head_dim + rope_dim_range[None, :]) * stride_dq_dim,
            mask=(offs_seq[:, None] < seq_len) & (rope_dim_range[None, :] < q_head_dim),
        )

        dq0_data = dq_data[:, :BLOCK_ROPE_DIM // 2]
        dq1_data = dq_data[:, BLOCK_ROPE_DIM // 2:]

        do_q_data = tl.empty_like(dq_data)
        do_q_data[:, :BLOCK_ROPE_DIM // 2] = (dq1_data * sin1_data + dq0_data * cos0_data).to(tl.bfloat16)
        do_q_data[:, BLOCK_ROPE_DIM // 2:] = (dq1_data * cos1_data - dq0_data * sin0_data).to(tl.bfloat16)
        do_q_data_view = tl.view(do_q_data, [BLOCK_SEQ, BLOCK_ROPE_DIM // 2, 2])
        do_q_data_view = tl.trans(do_q_data_view, 2, 0, 1)
        tl.store(
            d_q_out + offs_dq + (nope_head_dim + rope_dim_range[None, :]) * stride_dq_dim,
            do_q_data,
            mask=(offs_seq[:, None] < seq_len)
            & (rope_dim_range[None, :] < q_head_dim),
        )

        # for do_k
        offs_do_kv = (batch_idx * stride_k_nope_in_bsz + offs_seq[:, None] * stride_k_nope_in_seq +
                      head_idx * stride_k_nope_in_head)

        d_k_nope_data = tl.load(
            d_k_out + offs_d_kv + nope_dim_range[None, :] * stride_dk_out_dim,
            mask=(offs_seq[:, None] < seq_len) & (nope_dim_range[None, :] < nope_head_dim),
        )

        tl.store(
            d_k_nope_in + offs_do_kv + nope_dim_range[None, :],
            d_k_nope_data,
            mask=(offs_seq[:, None] < seq_len) & (nope_dim_range[None, :] < nope_head_dim),
        )


class PartialRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx: torch.Any, q, k_pe, k_nope, cos, sin):
        assert (q.is_contiguous() and k_pe.is_contiguous() and k_nope.is_contiguous() and cos.is_contiguous()
                and sin.is_contiguous())
        bsz, seq_len, num_heads, q_head_dim = q.shape
        assert bsz == k_pe.shape[0] and seq_len == k_pe.shape[1] and 1 == k_pe.shape[2]
        qk_rope_head_dim = k_pe.shape[3]
        assert qk_rope_head_dim == triton.next_power_of_2(qk_rope_head_dim)
        qk_nope_head_dim = q_head_dim - qk_rope_head_dim

        assert (bsz == k_nope.shape[0] and seq_len == k_nope.shape[1] and num_heads == k_nope.shape[2])
        assert (bsz == cos.shape[0] and seq_len == cos.shape[1] and qk_rope_head_dim == cos.shape[2])
        assert cos.shape == sin.shape
        stride_q_bsz, stride_q_seq, stride_q_head, stride_q_dim = q.stride()
        stride_kpe_bsz, stride_kpe_seq, stride_kpe_head, stride_kpe_dim = k_pe.stride()
        stride_cos_bsz, stride_cos_seq, stride_cos_dim = cos.stride()
        out_k = k_nope.new_empty(bsz, seq_len, num_heads, q_head_dim)
        with torch.cuda.device(q.device):
            grid = lambda META: (
                bsz,
                triton.cdiv(seq_len, META['BLOCK_SEQ']),
                triton.cdiv(num_heads, META['BLOCK_HEAD']),
            )
            _partial_rotary_emb_fwd_kernel[grid](
                q,
                k_pe,
                k_nope,
                cos,
                sin,
                out_k,
                stride_q_bsz,
                stride_q_seq,
                stride_q_head,
                stride_q_dim,
                stride_kpe_bsz,
                stride_kpe_seq,
                stride_kpe_head,
                stride_kpe_dim,
                stride_cos_bsz,
                stride_cos_seq,
                stride_cos_dim,
                stride_okv_bsz=out_k.stride(0),
                stride_okv_seq=out_k.stride(1),
                stride_okv_head=out_k.stride(2),
                stride_okv_dim=out_k.stride(3),
                stride_kv_bsz=k_nope.stride(0),
                stride_kv_seq=k_nope.stride(1),
                stride_kv_head=k_nope.stride(2),
                stride_kv_dim=k_nope.stride(3),
                seq_len=seq_len,
                q_head_dim=q_head_dim,
                rope_head_dim=qk_rope_head_dim,
                nope_head_dim=qk_nope_head_dim,
                num_heads=num_heads,
                BLOCK_ROPE_DIM=triton.next_power_of_2(qk_rope_head_dim),
                BLOCK_NOPE_DIM=triton.next_power_of_2(qk_nope_head_dim),
            )

        ctx.save_for_backward(k_nope, cos, sin)
        return q, out_k

    @staticmethod
    def backward(ctx, d_q, d_k_out):
        k_nope, cos, sin = ctx.saved_tensors
        bsz, seq_len, num_heads, q_head_dim = d_q.shape
        rope_head_dim = cos.shape[-1]
        nope_head_dim = q_head_dim - rope_head_dim
        d_k_pe_in = d_q.new_empty(bsz, seq_len, 1, rope_head_dim)
        d_k_nope = torch.empty_like(k_nope)
        d_q = d_q.contiguous()
        d_k_out = d_k_out.contiguous()

        with torch.cuda.device(d_q.device):
            grid = lambda META: (
                bsz,
                triton.cdiv(seq_len, META['BLOCK_SEQ']),
                triton.cdiv(num_heads, META['BLOCK_HEAD']),
            )
            _partial_rotary_emb_bwd_kernel[grid](
                d_q,
                d_k_out,
                cos,
                sin,
                d_k_pe_in,
                d_k_nope,
                stride_dq_bsz=d_q.stride(0),
                stride_dq_seq=d_q.stride(1),
                stride_dq_head=d_q.stride(2),
                stride_dq_dim=d_q.stride(3),
                stride_dk_out_bsz=d_k_out.stride(0),
                stride_dk_out_seq=d_k_out.stride(1),
                stride_dk_out_head=d_k_out.stride(2),
                stride_dk_out_dim=d_k_out.stride(3),
                stride_cos_bsz=cos.stride(0),
                stride_cos_seq=cos.stride(1),
                stride_cos_dim=cos.stride(2),
                stride_dokpe_bsz=d_k_pe_in.stride(0),
                stride_dokpe_seq=d_k_pe_in.stride(1),
                stride_dokpe_head=d_k_pe_in.stride(2),
                stride_dokpe_dim=d_k_pe_in.stride(3),
                stride_k_nope_in_bsz=k_nope.stride(0),
                stride_k_nope_in_seq=k_nope.stride(1),
                stride_k_nope_in_head=k_nope.stride(2),
                stride_k_nope_in_dim=k_nope.stride(3),
                seq_len=seq_len,
                q_head_dim=q_head_dim,
                rope_head_dim=rope_head_dim,
                nope_head_dim=nope_head_dim,
                BLOCK_ROPE_DIM=triton.next_power_of_2(rope_head_dim),
                BLOCK_NOPE_DIM=triton.next_power_of_2(nope_head_dim),
            )
        return d_q, d_k_pe_in, d_k_nope, None, None


def call(q, k_pe, kv, cos, sin):
    return PartialRotaryEmb.apply(q, k_pe, kv, cos, sin)


def bench_fn(q, k_pe, kv, cos, sin):
    fn = lambda: call(q, k_pe, kv, cos, sin)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = 'partial_rotary_emb'
for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for device_ in ['cuda']:
        num_heads = SymVar('num_heads')
        qk_nope_head_dim = SymVar('qk_nope_head_dim')
        qk_rope_head_dim = SymVar('qk_rope_head_dim')
        v_head_dim = SymVar('v_head_dim')
        q_head_dim = SymVar('q_head_dim')
        bsz, q_len = SymVar('bsz'), SymVar('q_len')
        # we dont' actually allocate tensor
        q = Tensor((bsz, q_len, num_heads, q_head_dim), dtype=dtype, device=device_)
        k_pe = Tensor(
            (bsz, q_len, SymVar('one'), qk_rope_head_dim),
            dtype=dtype,
            device=device_,
        )
        kv = Tensor(
            (bsz, q_len, num_heads, SymVar('qk_nope_head_dim + v_head_dim')),
            dtype=dtype,
            device=device_,
        )
        cos = Tensor((bsz, q_len, qk_rope_head_dim), dtype=dtype, device=device_)
        sin = Tensor((bsz, q_len, qk_rope_head_dim), dtype=dtype, device=device_)
        # space = ChoiceSpace([])
        register_dlblas_op(name, None, (q, k_pe, kv, cos, sin), call, bench_fn, call)
