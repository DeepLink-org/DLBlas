# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_rms_norm.py
# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import triton

from dlblas.kernels.rms_norm import rms_norm
from dlblas.utils.device_utils import infer_device


def _gt(input, weight, eps):
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


def test_rms_norm(input, weight, eps):
    return rms_norm(input, weight, eps)


def test():
    dtype = torch.float16
    device = infer_device()
    input = torch.rand(4, 8, dtype=dtype, device=device)
    weight = torch.rand(8, dtype=dtype, device=device)
    eps = 1e-6

    gt = _gt(input, weight, eps)
    tt = test_rms_norm(input, weight, eps)

    print('max diff', (gt - tt).abs().max())

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
            ms = triton.testing.do_bench(lambda: rms_norm(input, weight, eps), warmup=warmup, rep=rep)
        if 'pytorch' in provider:
            ms = triton.testing.do_bench(lambda: _gt(input, weight, eps), warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
