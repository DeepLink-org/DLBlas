# Copyright (c) 2025, DeepLink.
import os

import torch
import triton
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


def test_memory(func, _iter):
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated() / (2**20)
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['BT'],
        x_vals=[2**i for i in range(10, 13)],  # 1024, 2048, 4096
        xlabel='B x T',
        line_arg='provider',
        line_vals=['liger', 'huggingface'],
        line_names=['Liger', 'Hugging Face'],
        styles=[
            ('blue', 'solid'),
            ('orange', 'solid'),
        ],
        ylabel='GPU memory usage (MB)',
        plot_name='fused-linear-cross-entropy-memory-benchmark',
        args={
            'H': 4096,
            'V': 128256,
            'dtype': torch.float32
        },
    )
])
def bench_memory_cross_entropy(BT, H, V, provider, dtype, device='mlu'):
    print(f"Running benchmark with BT={BT}, H={H}, V={V}, dtype={dtype} provider={provider}")
    torch_lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        if provider == 'liger':
            return liger_lm_head_ce(_input, target)
        elif provider == 'huggingface':
            return torch_lm_head_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem = test_memory(full, _iter=10, provider=provider)
    return mem


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['BT'],
        x_vals=[2**i for i in range(10, 13)],  # 1024, 2048, 4096
        xlabel='B x T',
        line_arg='provider',
        line_vals=['liger', 'huggingface'],
        line_names=['Liger', 'Hugging Face'],
        styles=[
            ('blue', 'solid'),
            ('orange', 'solid'),
        ],
        ylabel='Time (ms)',
        plot_name='fused-linear-cross-entropy-speed-benchmark',
        args={
            'H': 4096,
            'V': 128256,
            'dtype': torch.float32
        },
    )
])
def bench_speed_cross_entropy(BT, H, V, provider, dtype, device='mlu'):
    print(f"Running benchmark with BT={BT}, H={H}, V={V}, dtype={dtype} provider={provider}")
    torch_lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        if provider == 'liger':
            return liger_lm_head_ce(_input, target)
        elif provider == 'huggingface':
            return torch_lm_head_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=quantiles, rep=100)
    return ms, min_ms, max_ms


bench_speed_cross_entropy.run(show_plots=True, print_data=True)
# bench_memory_cross_entropy.run(show_plots=True, print_data=True)
