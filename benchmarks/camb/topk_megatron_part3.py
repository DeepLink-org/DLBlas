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


def get_aux(scores_for_aux, mask_ce, num_experts, moe_aux_loss_coeff):
    ce = mask_ce.float().mean(0)
    Pi = scores_for_aux.mean(0)
    fi = ce * num_experts
    l_aux = (Pi * fi).sum() * moe_aux_loss_coeff

    return l_aux


def test():
    seq_len = 4096
    q_head_dim = 64
    num_experts = 64
    moe_aux_loss_coeff = 1
    mask_ce = torch.randint(0, num_experts, size=(seq_len, q_head_dim), dtype=torch.int32, device=device_)
    scores_for_aux = torch.randn(size=(seq_len, q_head_dim), dtype=torch.float, device=device_)

    with torch.no_grad():
        scores_for_aux_tri = (scores_for_aux.clone())
        mask_ce_tri = (mask_ce.clone())
    scores_for_aux.requires_grad = True
    scores_for_aux_tri.requires_grad = True

    l_aux = get_aux(scores_for_aux, mask_ce, num_experts, moe_aux_loss_coeff)

    from dlblas.kernels.camb.topk_megatron_part3 import topk_part3
    l_aux_tri = topk_part3.apply(scores_for_aux_tri, mask_ce_tri, num_experts, moe_aux_loss_coeff)
    print(f"out max diff: {(l_aux - l_aux_tri).abs().max().item()}")
    assert torch.allclose(l_aux, l_aux_tri, atol=1e-3, rtol=1e-3)

    # loss_torch = torch.sum(torch.mean(topk_weight_norm))
    # loss_torch.backward(retain_graph=True)
    # loss_tri = torch.sum(torch.mean(topk_weight_norm_tri))
    # loss_tri.backward(retain_graph=True)
    # print(f"out max diff: {(topk_weight.grad - topk_weight_tri.grad).abs().max().item()}")

    # assert torch.allclose(topk_weight.grad, topk_weight_tri.grad,atol=1e-3,rtol=1e-3)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['pytorch', 'triton'],
            line_names=['PyTorch', 'Triton'],
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
                fn = lambda: topk_part3.apply(scores_for_aux_tri, mask_ce_tri, num_experts, moe_aux_loss_coeff)
            # elif "bwd" == op:
            #     topk_weight_norm_tri = topk_part2.apply(topk_weight_tri)
            #     loss_tri = torch.sum(topk_weight_norm_tri)
            #     fn = lambda: loss_tri.backward(retain_graph=True)
        if 'pytorch' in provider:
            if 'fwd' == op:
                fn = lambda: get_aux(scores_for_aux, mask_ce, num_experts, moe_aux_loss_coeff)
            elif 'bwd' == op:
                l_aux = get_aux(scores_for_aux, mask_ce, num_experts, moe_aux_loss_coeff)
                loss_torch = torch.sum(l_aux)
                fn = lambda: loss_torch.backward(retain_graph=True)
            else:
                raise Exception()
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
    print('sucessfully!')
