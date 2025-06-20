# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_fill_kv_cache.py
import torch
import triton

# import torch_mlu
# import torch_mlu.utils.gpu_migration
from dlblas.kernels.fill_kv_cache import fill_kv_cache
from dlblas.utils.device_utils import infer_device

device = infer_device()


def _div_up(a, b):
    return (a + b - 1) // b


def _block_offsets(num_blocks_per_input):
    batch_size = len(num_blocks_per_input)
    max_num_blocks = max(num_blocks_per_input)
    batch_ids = torch.arange(batch_size)
    ret = torch.arange(max_num_blocks)
    ret = batch_ids[:, None] + ret[None, :] * batch_size
    return ret.to(device)


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
    batch_size = len(seq_lens)
    k_caches = k_caches.clone()
    v_caches = v_caches.clone()
    splited_k_states = k_states.split(seq_lens)
    splited_v_states = v_states.split(seq_lens)
    for bidx in range(batch_size):
        k_state = splited_k_states[bidx]
        v_state = splited_v_states[bidx]
        h_len = history_lens[bidx]
        b_offs = block_offsets[bidx]
        block_id = _div_up(h_len + 1, block_size) - 1
        fill_start = h_len % block_size
        fill_size = min(block_size - fill_start, k_state.size(0))
        while True:
            boff = b_offs[block_id]
            tmp_ks = k_state[:fill_size]
            tmp_vs = v_state[:fill_size]
            fill_end = fill_start + fill_size
            k_caches[boff, fill_start:fill_end] = tmp_ks
            v_caches[boff, fill_start:fill_end] = tmp_vs
            k_state = k_state[fill_size:]
            v_state = v_state[fill_size:]
            block_id += 1
            fill_start = 0
            fill_size = min(block_size, k_state.size(0))
            if fill_size == 0:
                break

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
    num_heads = 8
    head_dim = 128
    block_size = 4
    seq_lens = [10]
    history_lens = [10]

    batch_size = len(seq_lens)
    kv_lens = [s + h for s, h, in zip(seq_lens, history_lens)]
    max_q_seq_length = max(seq_lens)
    num_tokens = sum(seq_lens)
    num_blocks_per_input = [_div_up(kv_len, block_size) for kv_len in kv_lens]
    max_num_blocks = max(num_blocks_per_input)
    q_seq_length = torch.tensor(seq_lens).to(device)
    q_start_loc = q_seq_length.cumsum(0) - q_seq_length
    kv_seq_length = torch.tensor(kv_lens).to(device)
    k_states = torch.rand(num_tokens, num_heads, head_dim).to(device)
    v_states = torch.rand_like(k_states)
    k_caches = torch.full((batch_size * max_num_blocks, block_size, num_heads, head_dim), 0.0).to(device)
    v_caches = torch.rand_like(k_caches)
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
    def bench_fn(op, provider):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            ms = triton.testing.do_bench(lambda: fill_kv_cache(
                k_states,
                v_states,
                k_caches,
                v_caches,
                q_start_loc,
                q_seq_length,
                kv_seq_length,
                max_q_seq_length,
                block_offsets,
            ),
                                         warmup=warmup,
                                         rep=rep)
        if 'pytorch' in provider:
            ms = triton.testing.do_bench(lambda: _gt(
                k_states,
                v_states,
                k_caches,
                v_caches,
                seq_lens,
                history_lens,
                block_offsets,
                block_size,
            ),
                                         warmup=warmup,
                                         rep=rep)

        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
