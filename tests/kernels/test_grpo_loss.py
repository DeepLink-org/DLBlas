# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_rms_norm.py
import pytest
import torch
import os
from typing import Callable, Optional, List

from dlblas.kernels.grpo_loss import grpo_loss_forward, grpo_loss_backward


def benchmark_with_event(
    target_fn: Callable[[None], None],
    warmup_iters: int = 10,
    benchmark_iters: int = 20,
    profile_ranks: Optional[List[int]] = None,
    flush_l2: bool = False,
    cuda_graph: bool = False,
) -> float:
    if cuda_graph:
        target_fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            target_fn()
        target_fn = lambda: g.replay()

    if "BENCHMARK_ITERS" in os.environ:
        benchmark_iters = int(os.environ["BENCHMARK_ITERS"])

    rank = 0  # dist.get_rank() if dist.is_initialized() else 0
    profile_ranks = profile_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup_iters):
        target_fn()

    torch.cuda.synchronize()

    begin_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    if rank in profile_ranks:
        try:
            from trace_handler import trace_handler
        except ImportError:
            trace_handler = None

        if "NO_TRACE" in os.environ:
            trace_handler = None

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
        )
    else:
        prof = nullcontext()

    with prof:
        torch.cuda._sleep(int(2e7))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            begin_events[i].record()
            target_fn()
            end_events[i].record()
        torch.cuda.synchronize()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]
    return torch.tensor(latencies).median().item() * 1000


def reference_forward(_logprobs, _old_logprobs, _advantages, _ref_logprobs, kl_type=1, kl_coef=1.0, _loss_factor = 1.0, clip = 0.2):
    logprobs_diff = _logprobs - _old_logprobs
    ratio = torch.exp(logprobs_diff)
    pg_losses = -_advantages.unsqueeze(1) * ratio
    pg_losses2 = -_advantages.unsqueeze(1) * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    pg_loss_max = torch.max(pg_losses, pg_losses2)
    pg_loss = pg_loss_max.sum()
    _loss = pg_loss * _loss_factor

    # Compute KL penalty loss
    if kl_type == 0:
        kl = _ref_logprobs - _logprobs 
        _kl_penalty_loss = (kl_coef * kl).sum(dim=1) * _loss_factor  
    elif kl_type == 1:
        kl = _ref_logprobs - _logprobs
        nonneg_nobias_kl = torch.exp(kl) - kl - 1
        _kl_penalty_loss = (kl_coef * nonneg_nobias_kl).sum(dim=1) * _loss_factor
    elif kl_type == 2:
        _kl_penalty_loss = (kl_coef * (_ref_logprobs - _logprobs).square() / 2).sum(dim=1) * _loss_factor
    else:
        raise ValueError(f"Unsupported KL type: {kl_type}")
    loss = _loss + _kl_penalty_loss
    return loss


def reference_backward(log_probs,  ref_loss):
    ref_loss.backward(ref_loss)
    ref_logp = log_probs.grad
    return ref_logp


class TestGRPOLoss:

    def test_grpo_loss(self):

        B = 8
        T = 32
        H = 256
        V = 1024
        BLOCK_SIZE_T = 32

        torch.cuda.set_device('cuda:0')
        torch.manual_seed(42)

        loss = torch.zeros((T,), dtype=torch.float32, device='cuda', requires_grad=True)
        advantages = torch.randn((T,), dtype=torch.float32, device='cuda', requires_grad=True)
        log_probs = torch.randn((T, V), dtype=torch.float32, device='cuda', requires_grad=True)
        log_probs1 = torch.randn((T, V), dtype=torch.float32, device='cuda', requires_grad=True)
        out_logprobs = torch.empty((T, V), dtype=torch.float32, device='cuda', requires_grad=True)

        loss_factor = kl_coef = 1.0
        kl_type = "unbias"
        clip = 0.2

        lat_tri_loss = benchmark_with_event(
            lambda: grpo_loss_forward(log_probs, log_probs1, log_probs1,
                            advantages, kl_type, kl_coef, loss_factor, clip,
                            loss, B, T, V, BLOCK_SIZE_T),
            flush_l2=True,
        )

        tri_loss = grpo_loss_forward(log_probs, log_probs1, log_probs1,
                            advantages, kl_type, kl_coef, loss_factor, clip,
                            loss, B, T, V, BLOCK_SIZE_T)

        lat_out_logp = benchmark_with_event(
            lambda: grpo_loss_backward(tri_loss, log_probs, log_probs1, log_probs1,
                        out_logprobs, advantages, clip, B, T, V, BLOCK_SIZE_T),
            flush_l2=True,
            warmup_iters=0,
            benchmark_iters=1,
        )

        out_logp = grpo_loss_backward(tri_loss, log_probs, log_probs1, log_probs1,
                        out_logprobs, advantages, clip, B, T, V, BLOCK_SIZE_T)

        lat_ref_loss = benchmark_with_event(
            lambda: reference_forward(log_probs, log_probs1, advantages,
                    log_probs1, 1, kl_coef, loss_factor, clip),
            flush_l2=True,
        )

        ref_loss = reference_forward(log_probs, log_probs1, advantages,
                    log_probs1, 1, kl_coef, loss_factor, clip)

        lat_ref_logp = benchmark_with_event(
            lambda: reference_backward(log_probs, ref_loss),
            flush_l2=True,
            warmup_iters=0,
            benchmark_iters=1,
        )

        assert torch.allclose(tri_loss, ref_loss)
        assert torch.allclose(out_logp, log_probs.grad)

        print('forward accuracy: ', torch.allclose(tri_loss, ref_loss))
        print('backward accuracy: ', torch.allclose(out_logp, log_probs.grad))

        print('tri_loss latency: ', lat_tri_loss)
        print('ref_loss latency: ', lat_ref_loss)
        print('tri_logp latency: ', lat_out_logp)
        print('ref_logp latency: ', lat_ref_logp)
