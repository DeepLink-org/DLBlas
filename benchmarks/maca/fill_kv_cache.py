# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_fill_kv_cache.py
import pytest
import torch
import triton
import vllm

# import torch_mlu
# import torch_mlu.utils.gpu_migration
from dlblas.kernels.fill_kv_cache import fill_kv_cache


def _div_up(a, b):
    return (a + b - 1) // b


def _block_offsets(num_blocks_per_input):
    batch_size = len(num_blocks_per_input)
    max_num_blocks = max(num_blocks_per_input)
    batch_ids = torch.arange(batch_size)
    ret = torch.arange(max_num_blocks)
    ret = batch_ids[:, None] + ret[None, :] * batch_size
    return ret.cuda()


def _gt(
    k_states,
    v_states,
    k_caches,
    v_caches,
    seq_lens,
    history_lens,
    block_offsets,
    block_size,
):
    k_caches = k_caches.clone()
    v_caches = v_caches.clone()
    kv_indices = torch.arange(k_states.shape[0]).cuda()
    vllm._custom_ops.reshape_and_cache_new(k_states, v_states, k_caches, v_caches, kv_indices, 'auto', 1.0, 1.0)

    return k_caches, v_caches


def _test_fill_kv_cache(
    k_states,
    v_states,
    k_caches,
    v_caches,
    block_offsets,
    q_start_loc,
    q_seq_length,
    kv_seq_length,
    max_q_seq_length,
):
    fill_kv_cache(
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        max_q_seq_length,
        block_offsets,
    )


def test():
    num_heads = 10
    head_dim = 128
    block_size = 8
    seq_lens = [10]
    history_lens = [10]
    x = 16

    batch_size = len(seq_lens)
    kv_lens = [s + h for s, h, in zip(seq_lens, history_lens)]
    max_q_seq_length = max(seq_lens)
    num_tokens = sum(seq_lens)
    num_blocks_per_input = [_div_up(kv_len, block_size) for kv_len in kv_lens]
    max_num_blocks = max(num_blocks_per_input)
    q_seq_length = torch.tensor(seq_lens).cuda()
    q_start_loc = q_seq_length.cumsum(0) - q_seq_length
    kv_seq_length = torch.tensor(kv_lens).cuda()
    k_states = torch.rand(num_heads, block_size, head_dim).cuda()
    v_states = torch.rand_like(k_states)
    k_caches = torch.full((130, num_heads, head_dim // x, block_size, x), 0.0).cuda()
    v_caches = torch.rand_like(k_caches).reshape((130, block_size, num_heads, head_dim))
    block_offsets = _block_offsets(num_blocks_per_input)

    gt = _gt(
        k_states,
        v_states,
        k_caches,
        v_caches,
        seq_lens,
        history_lens,
        block_offsets,
        block_size,
    )
    tt = fill_kv_cache(
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        max_q_seq_length,
        block_offsets,
    )

    print('k_cache max diff', (gt[0] - k_caches).abs().max())
    print('v_cache max diff', (gt[1] - v_caches).abs().max())

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
            fn = lambda: fill_kv_cache(
                k_states,
                v_states,
                k_caches,
                v_caches,
                q_start_loc,
                q_seq_length,
                kv_seq_length,
                max_q_seq_length,
                block_offsets,
            )
        if 'pytorch' in provider:
            fn = lambda: _gt(
                k_states,
                v_states,
                k_caches,
                v_caches,
                seq_lens,
                history_lens,
                block_offsets,
                block_size,
            )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
