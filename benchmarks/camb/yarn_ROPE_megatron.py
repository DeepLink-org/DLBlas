# Copyright (c) 2025, DeepLink.
import time
from typing import List

import torch
import torch.nn.functional as F
import torch_mlu
import triton
from functorch.compile import aot_function, aot_module, make_boxed_func
from torch import nn
from torch._dynamo.backends.common import aot_autograd
from torch.profiler import ProfilerActivity, profile, record_function
from torch_mlu.utils.model_transfer import transfer

import dlblas
from dlblas.utils.device_utils import get_idle_device

device_ = torch.device(get_idle_device())
torch.cuda.set_device(device_)


def yarn_ROPE(max_seq_len, offset, inv_freq):
    seq = (torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype) + offset)

    freqs = torch.outer(seq, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    # # emb [seq_length, .., dim]
    return emb


def test():
    max_seq_len = 4096
    q_head_dim = 32
    offset = 0
    inv_freq = torch.arange(0, q_head_dim, dtype=torch.float32, device=device_)

    with torch.no_grad():
        inv_freq_tri = (inv_freq.clone())
    inv_freq.requires_grad = True
    inv_freq_tri.requires_grad = True

    emb = yarn_ROPE(max_seq_len, offset, inv_freq)
    loss_torch = torch.sum(torch.mean(emb))
    loss_torch.backward(retain_graph=True)

    from dlblas.kernels.camb.yarn_ROPE import yarnROPE
    emb_triton = yarnROPE.apply(max_seq_len, offset, inv_freq_tri)

    print(f"out_q max diff: {(emb - emb_triton).abs().max().item()}")
    assert torch.allclose(emb, emb_triton)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            #line_vals=["pytorch"],
            #line_names=["PyTorch"],
            ylabel='ms',
            plot_name=f"yarn_ROPE",
            args={'q_head_dim': q_head_dim},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(q_head_dim, op, provider, device=device_):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            if 'fwd' == op:
                fn = lambda: yarnROPE.apply(max_seq_len, offset, inv_freq)

        if 'pytorch' in provider:
            if 'fwd' == op:
                fn = lambda: yarn_ROPE(max_seq_len, offset, inv_freq)
            # elif "bwd" == op:
            #     out_q, out_k = partial_rotary_emb(q, k_pe, k_nope, cos, sin)
            #     loss_torch = torch.sum(torch.mean(out_q) * torch.mean(out_k))
            #     fn = lambda: loss_torch.backward(retain_graph=True)
            # else:
            #     raise Exception()
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print('sucessfully!')
