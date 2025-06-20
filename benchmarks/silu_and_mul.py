# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_activation.py
import pytest
# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import triton

from dlblas.kernels.activation import silu_and_mul
from dlblas.utils.device_utils import infer_device

DEVICE = infer_device()


class TestSiluAndMul:

    @pytest.fixture
    def seqlen(self):
        yield 256

    @pytest.fixture
    def feat_size(self, request):
        yield request.param

    @pytest.fixture
    def x(self, seqlen, feat_size):
        yield torch.rand(seqlen, feat_size, dtype=torch.float16, device=DEVICE)

    @pytest.fixture
    def gt(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        yield gate * up

    @pytest.mark.parametrize('feat_size', [4096, 768], indirect=True)
    def test_silu_and_mul(self, x, gt):
        out = silu_and_mul(x)
        torch.testing.assert_close(out, gt)


def _gt(x):
    gate, up = x.chunk(2, -1)
    gate = torch.nn.functional.silu(gate)
    return gate * up


def _test_silu_and_mul(x):
    return silu_and_mul(x)


def test():
    seqlen = 256
    feat_size = 4096
    x = torch.rand(seqlen, feat_size, dtype=torch.float16, device=DEVICE)

    gt = _gt(x)
    tt = _test_silu_and_mul(x)

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
            ms = triton.testing.do_bench(lambda: silu_and_mul(x), warmup=warmup, rep=rep)
        if 'pytorch' in provider:
            ms = triton.testing.do_bench(lambda: _gt(x), warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
