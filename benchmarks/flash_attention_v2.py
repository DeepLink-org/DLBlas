# Copyright (c) 2025, DeepLink.
import torch
import torch.nn.functional as F
import triton

from dlblas.kernels.flash_attention_v2 import _flash_attn_forward as flash_attention_v2
from dlblas.utils.device_utils import infer_device, is_cuda, is_muxi

MUXI_CUDA = is_muxi() or is_cuda()


def test():
    device_ = torch.device(infer_device())
    dtype = torch.bfloat16
    # if MUXI_CUDA:
    #     dtype = torch.float32

    seq_len, heads, dim = 25600, 32, 96  #if dim=96,using hdim96 kernel
    query = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    key = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)
    value = torch.rand([1, seq_len, heads, dim], dtype=dtype, device=device_)

    tt_out, _, _ = flash_attention_v2(query, key, value)
    ref_out = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)

    print('TEST: ')
    tt_out = tt_out
    print('max abs diff: ', torch.max(abs(tt_out - ref_out)))
    assert torch.allclose(tt_out, ref_out, atol=1e-2, rtol=0)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['triton', 'torch'],
            ylabel='ms',
            plot_name=f"flashAttention(batchSize={1}, seqlen={seq_len}, num_heads={heads}, dim={dim})",
            args={'SeqLen': seq_len},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(SeqLen, op, provider, device=device_):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            ms = triton.testing.do_bench(lambda: flash_attention_v2(query, key, value), warmup=warmup, rep=rep)

        if 'torch' in provider:
            ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
            ).permute(0, 2, 1, 3),
                                         warmup=warmup,
                                         rep=rep)

        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print('sucessfully!')
