import math

import pytest
# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import triton

from dlblas.kernels.paged_attention import paged_attention_fwd


def _conti_input(data, seq_lens):
    data = [x[:l] for x, l in zip(data, seq_lens)]
    data = torch.cat(data, dim=0)
    return data


def _make_bias(seq_lens, history_lens, neg_val):
    full_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()
    max_full_len = full_seq_lens.max().item()
    seq_ranges = [torch.arange(max_seq_len) for _ in seq_lens]
    for r, l in zip(seq_ranges, seq_lens):
        r[l:] = -max_full_len
    seq_ranges = torch.stack(seq_ranges, dim=0).cuda()
    kv_ranges = [torch.arange(max_full_len) for _ in full_seq_lens]
    kv_ranges = torch.stack(kv_ranges, 0).cuda()
    mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None]
    return mask.float() * neg_val


def _make_blocked_cache(
    batched_k,
    batched_v,
    seq_lens,
    history_lens,
    block_offsets,
    block_size,
    num_heads_k,
    feat_dim,
    feat_dim_v,
):
    max_blocks_nums = block_offsets.max() + 1
    full_seq_lens = seq_lens + history_lens
    blocked_k = batched_k.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim)
    blocked_v = batched_v.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim_v)

    for batch_id, offset in enumerate(block_offsets):
        ori_k = batched_k[batch_id]
        ori_v = batched_v[batch_id]
        seq_len = full_seq_lens[batch_id]
        for block_id, block_start in enumerate(range(0, seq_len, block_size)):
            block_off = offset[block_id]
            tmp_k = ori_k[block_start:block_start + block_size]
            tmp_v = ori_v[block_start:block_start + block_size]
            size = tmp_k.size(0)
            blocked_k[block_off, :size] = tmp_k
            blocked_v[block_off, :size] = tmp_v

    return blocked_k, blocked_v


def _naive_attention(batched_q, batched_kv, bias):
    batched_k, batched_v = batched_kv

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k

    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    # expand group
    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    attn_weight = qk + bias[:, None]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


def _naive_window_attention(q, k, v, seqlens_q, seqlens_k, window_size):
    from flash_attn import flash_attn_varlen_func

    def _make_cu_seqlens(seqlens):
        cu_seqlens = seqlens.cumsum(0)
        cu_zero = cu_seqlens.new_zeros(1)
        cu_seqlens = torch.cat([cu_zero, cu_seqlens])
        return cu_seqlens

    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    cu_seqlens_q = _make_cu_seqlens(seqlens_q).int()
    cu_seqlens_k = _make_cu_seqlens(seqlens_k).int()

    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        window_size=window_size,
    )
    return output


def _start_loc(seq_lens):
    seq_sum = seq_lens.cumsum(0)
    start_loc = torch.cat([seq_sum.new_zeros(1), seq_sum[:-1]], dim=0)
    return start_loc


def _batched_q(seq_lens, num_heads_q, feat_dim, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    max_seq_len = seq_lens.max().item()
    return torch.randn(batch_size, max_seq_len, num_heads_q, feat_dim, dtype=dtype, device='cuda')


def _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    full_seq_lens = seq_lens + history_lens
    max_seq_len = full_seq_lens.max().item()
    k = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device='cuda')
    v = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device='cuda')
    return k, v


def _conti_q(seq_lens, batched_q):
    return _conti_input(batched_q, seq_lens)


def _block_offsets(seq_lens, history_lens, block_size):
    full_seq_lens = seq_lens + history_lens
    batch_size = full_seq_lens.size(0)
    num_blocks = (full_seq_lens + block_size - 1) // block_size

    offset = [torch.arange(size) * batch_size + idx for idx, size in enumerate(num_blocks)]
    max_len = max(len(o) for o in offset)
    new_offset = offset[0].new_zeros(batch_size, max_len)
    for o, no in zip(offset, new_offset):
        len_o = o.size(0)
        no[:len_o] = o

    print(new_offset)

    return new_offset.cuda()


def _conti_kv(batched_kv, seq_lens, history_lens):
    full_seq_lens = seq_lens + history_lens
    conti_k = _conti_input(batched_kv[0], full_seq_lens)
    conti_v = _conti_input(batched_kv[1], full_seq_lens)
    return (conti_k, conti_v)


def _blocked_kv(
    batched_kv,
    seq_lens,
    history_lens,
    block_offsets,
    block_size,
    num_heads_k,
    feat_dim,
    feat_dim_v,
):
    batched_k, batched_v = batched_kv
    return _make_blocked_cache(
        batched_k,
        batched_v,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
        num_heads_k,
        feat_dim,
        feat_dim_v,
    )


def _mask(seq_lens, history_lens):
    neg_val = -1e30
    return _make_bias(seq_lens, history_lens, neg_val)


def _gt(batched_q, batched_kv, mask):
    return _naive_attention(batched_q, batched_kv, mask)


def _conti_gt(gt, seq_lens):
    return _conti_input(gt, seq_lens)


def test_paged_attention(
    conti_q,
    blocked_kv,
    block_offsets,
    start_loc,
    seq_lens,
    history_lens,
    feat_dim_v,
):
    kv_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()

    blocked_k, blocked_v = blocked_kv
    out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

    paged_attention_fwd(
        conti_q,
        blocked_k,
        blocked_v,
        out,
        block_offsets=block_offsets,
        q_start_loc=start_loc,
        q_seqlens=seq_lens,
        kv_seqlens=kv_seq_lens,
        max_seqlen=max_seq_len,
    )


def test():
    dtype = torch.float16
    feat_dim = 16
    feat_dim_v = 16
    num_heads_q = 4
    num_heads_k = 2
    seq_lens = torch.tensor([128], device='cuda')
    start_loc = _start_loc(seq_lens)
    block_size = 16
    history_lens = torch.tensor([128], device='cuda')
    win_size = 32

    batched_q = _batched_q(seq_lens, num_heads_q, feat_dim, dtype)
    batched_kv = _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype)
    conti_q = _conti_q(seq_lens, batched_q)
    block_offsets = _block_offsets(seq_lens, history_lens, block_size)
    conti_kv = _conti_kv(batched_kv, seq_lens, history_lens)
    blocked_kv = _blocked_kv(
        batched_kv,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
        num_heads_k,
        feat_dim,
        feat_dim_v,
    )
    mask = _mask(seq_lens, history_lens)
    # _conti_gt(_gt(batched_q, batched_kv, mask), seq_lens)
    kv_seq_lens = seq_lens + history_lens
    max_seq_len = seq_lens.max().item()

    blocked_k, blocked_v = blocked_kv
    out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)
    paged_attention_fwd(
        conti_q,
        blocked_k,
        blocked_v,
        out,
        block_offsets=block_offsets,
        q_start_loc=start_loc,
        q_seqlens=seq_lens,
        kv_seqlens=kv_seq_lens,
        max_seqlen=max_seq_len,
    )

    conti_gt = _conti_gt(_gt(batched_q, batched_kv, mask), seq_lens)

    print('max diff', (conti_gt - out).abs().max())
    torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            ylabel='ms',
            plot_name='',
            args={},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device='cuda'):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            # fn = lambda: test_paged_attention(conti_q, blocked_kv, block_offsets, start_loc, seq_lens, history_lens, feat_dim_v)
            fn = lambda: paged_attention_fwd(
                conti_q,
                blocked_k,
                blocked_v,
                out,
                block_offsets=block_offsets,
                q_start_loc=start_loc,
                q_seqlens=seq_lens,
                kv_seqlens=kv_seq_lens,
                max_seqlen=max_seq_len,
            )
        if 'pytorch' in provider:
            fn = lambda: _conti_gt(_gt(batched_q, batched_kv, mask), seq_lens)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
