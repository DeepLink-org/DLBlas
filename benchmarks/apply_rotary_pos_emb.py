import torch
import triton

import dlblas
from dlblas.kernels.apply_rotary_pos_emb import apply_rotary_pos_emb
from dlblas.utils.device_utils import get_idle_device


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def torch_rotary_pos_emb(q_states, k_states, cached_cos, cached_sin, position_ids_1d):
    cos = cached_cos[position_ids_1d, None, :]
    sin = cached_sin[position_ids_1d, None, :]
    q_embed = q_states * cos + _rotate_half(q_states) * sin
    k_embed = k_states * cos + _rotate_half(k_states) * sin
    return q_embed, k_embed


def test():
    device_ = torch.device(get_idle_device())
    torch.cuda.set_device(device_)
    dtype_ = torch.float16
    b, s, h, d = 1, 256, 32, 128
    cached_cos = torch.randn((s, d), dtype=dtype_, device=device_)
    cached_sin = torch.randn((s, d), dtype=dtype_, device=device_)
    q_states = torch.randn((b * s, h, d), dtype=dtype_, device=device_)
    k_states = torch.randn((b * s, h, d), dtype=dtype_, device=device_)
    # position_ids_1d = torch.randint(0, s, (b * s,), device=device_)
    position_ids_1d = torch.arange(0, s, device=device_)
    q_embed, k_embed = torch_rotary_pos_emb(q_states, k_states, cached_cos, cached_sin, position_ids_1d)
    q_embed_tri, k_embed_tri = apply_rotary_pos_emb(q_states, k_states, cached_cos, cached_sin)
    print('max abs diff: ', torch.max(abs(q_embed - q_embed_tri)))
    print('max abs diff: ', torch.max(abs(k_embed - k_embed_tri)))
    assert torch.allclose(q_embed, q_embed_tri, atol=1e-2, rtol=1e-1)
    assert torch.allclose(k_embed, k_embed_tri, atol=1e-2, rtol=1e-1)

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
            fn = lambda: apply_rotary_pos_emb(q_states, k_states, cached_cos, cached_sin)
        if 'pytorch' in provider:
            fn = lambda: torch_rotary_pos_emb(q_states, k_states, cached_cos, cached_sin, position_ids_1d)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':

    test()
    print('sucessfully!')
