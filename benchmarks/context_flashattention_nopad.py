# Copyright (c) 2025, DeepLink.
import math

import numpy as np
import torch
import torch.nn.functional as F
import triton

from dlblas.kernels.context_flashattention_nopad import context_attention_fwd_no_prompt_cache
from dlblas.utils.device_utils import get_idle_device


def torch_att(q, q_rope, kv, kv_rope, bs, seqlen, num_head, q_head_dim, rope_head_dim):

    xq = torch.cat([q, q_rope], dim=2).view(bs, seqlen, num_head, -1)
    xk = torch.cat([kv, kv_rope], dim=2).view(bs, seqlen, 1, -1)
    xv = kv.view(bs, seqlen, 1, -1)

    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.0] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(q_head_dim + rope_head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, q_head_dim)
    return output


def test():
    device_ = torch.device(get_idle_device())
    torch.cuda.set_device(device_)

    Z, H, N_CTX, D_HEAD, ROPE_HEAD = 1, 6, 5000, 128, 64
    dtype = torch.float16
    Z = 1
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0.3, std=0.2)
    q_rope = torch.empty((Z * N_CTX, H, ROPE_HEAD), dtype=dtype, device='cuda').normal_(mean=0.3, std=0.2)

    kv = torch.empty((Z * N_CTX, 1, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0.3, std=0.2)
    kv_rope = torch.empty((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device='cuda').normal_(mean=0.3, std=0.2)

    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0.7, std=0.2)
    o1 = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0.7, std=0.2)

    req_to_token_indexs = torch.zeros((10, Z * N_CTX), dtype=torch.int32, device='cuda')
    max_input_len = N_CTX
    Z = 1
    b_start_loc = torch.zeros((Z, ), dtype=torch.int32, device='cuda')
    b_seq_len = torch.ones((Z, ), dtype=torch.int32, device='cuda')
    b_req_idx = torch.ones((Z, ), dtype=torch.int32, device='cuda')
    b_prompt_cache_len = torch.zeros(1, dtype=torch.int32, device='cuda')
    b_prompt_cache_len[0] = 0
    prompt_cache_len = 0

    b_seq_len[0] = N_CTX
    b_req_idx[0] = 0
    req_to_token_indexs[0][:prompt_cache_len + N_CTX] = torch.tensor(np.arange(prompt_cache_len + N_CTX),
                                                                     dtype=torch.int32).cuda()

    softmax_scale = 1 / math.sqrt(D_HEAD + ROPE_HEAD)

    torch_out = torch_att(q, q_rope, kv, kv_rope, Z, N_CTX, H, D_HEAD, ROPE_HEAD)

    context_attention_fwd_no_prompt_cache(q, q_rope, kv, kv_rope, o, b_start_loc, b_seq_len, max_input_len,
                                          softmax_scale)

    print('TEST: ')
    print('max ', torch.max(torch.abs(torch_out - o)))
    print('mean ', torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['triton', 'torch'],
            ylabel='ms',
            plot_name=f"contextFlashAttention();  # Z, H, N_CTX, D_HEAD, ROPE_HEAD = 1, 6, 5000, 128, 64",
            args={},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device=device_):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            fn = lambda: context_attention_fwd_no_prompt_cache(q, q_rope, kv, kv_rope, o, b_start_loc, b_seq_len,
                                                               max_input_len, softmax_scale)

        if 'torch' in provider:
            fn = lambda: torch_att(q, q_rope, kv, kv_rope, Z, N_CTX, H, D_HEAD, ROPE_HEAD)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print('sucessfully!')
