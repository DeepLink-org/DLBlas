# Copyright (c) 2025, DeepLink.
import torch
import triton
import os
from typing import Callable, Optional, List
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.grpo_loss import GRPOLoss


KL = 0
UNBIAS = 1
MSE = 2
device_ = infer_device()

def torch_grpo_loss( _logprobs, _old_logprobs, _advantages, _ref_logprobs,
        kl_type=1, kl_coef=1.0, _loss_factor = 1.0, clip = 0.2):
    kl_type = {
        'kl': KL,
        'unbias': UNBIAS,
        'mse': MSE
    }.get(kl_type, None)

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


def triton_grpo_loss(
    log_probs, log_probs1, log_probs2,
    advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T
):
    assert log_probs is not None and log_probs1 is not None and log_probs2 is not None

    return GRPOLoss.apply(
        log_probs, log_probs1, log_probs2,
        advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T
    )


def all_close(name, tensor1, tensor2, rtol, atol):
    # 确保张量在相同设备上
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)

    # 计算各项差异指标
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor2) + 1e-8)  # 防止除以0

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    abs_diff_mean = abs_diff.mean().item()
    rel_diff_mean = rel_diff.mean().item()

    # 计算通过比例
    abs_pass = (abs_diff <= atol)
    rel_pass = (rel_diff <= rtol)
    pass_mask = (abs_pass | rel_pass)
    pass_rate = pass_mask.float().mean().item() * 100

    # 使用torch.allclose判断是否通过
    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    # 输出结果
    print(f"\n===== {name} 比较结果 =====")
    print(f"张量形状: {tensor1.shape} (原始), {tensor2.shape} (参考)")
    print(f"最大绝对误差: {max_abs_diff:.4e} | 最大相对误差: {max_rel_diff:.4e}")
    print(f"平均绝对误差: {abs_diff_mean:.4e} | 平均相对误差: {rel_diff_mean:.4e}")
    print(f"精度通过率: {pass_rate:.2f}% ({pass_mask.sum()}/{pass_mask.numel()} 元素)")
    print(f"符合精度要求(rtol={rtol}, atol={atol}): {'是' if is_close else '否'}")

    return is_close


def grpo_loss_kernel(kl_type="unbias"):
    B = 8
    T = 32
    H = 256
    V = 1024
    BLOCK_SIZE_T = 4
    torch.manual_seed(42)

    advantages = torch.randn((T,), dtype=torch.float32, device=device_, requires_grad=True)
    log_probs = torch.randn((T, V), dtype=torch.float32, device=device_, requires_grad=True)
    log_probs1 = torch.randn((T, V), dtype=torch.float32, device=device_, requires_grad=True)
    log_probs2 = torch.randn((T, V), dtype=torch.float32, device=device_, requires_grad=True)

    std_advantages = torch.tensor(advantages.detach(), requires_grad=True)
    std_log_probs = torch.tensor(log_probs.detach(), requires_grad=True)
    std_log_probs1 = torch.tensor(log_probs1.detach(), requires_grad=True)
    std_log_probs2 = torch.tensor(log_probs2.detach(), requires_grad=True)

    loss_factor = kl_coef = 1.0
    clip = 0.2
    tri_loss = triton_grpo_loss(log_probs, log_probs1, log_probs2,
                        advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T)
    tri_loss.backward(tri_loss)
    ref_loss = torch_grpo_loss(std_log_probs, std_log_probs1, std_advantages,
                std_log_probs2, kl_type, kl_coef, loss_factor, clip)
    ref_loss.backward(ref_loss)


    # # 性能测试部分
    # def benchmark_fn(fn, *args):
    #     return triton.testing.do_bench(lambda: fn(*args), warmup=100, rep=500)

    # # Triton 版本性能
    # tri_fwd_time = benchmark_fn(triton_grpo_loss, log_probs, log_probs1, log_probs2, 
    #                           advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T)
    # # tri_bwd_time = benchmark_fn(lambda: tri_loss.backward(tri_loss))

    # # PyTorch 版本性能
    # torch_fwd_time = benchmark_fn(torch_grpo_loss, std_log_probs, std_log_probs1, std_advantages,
    #                             std_log_probs2, kl_type, kl_coef, loss_factor, clip)
    # torch_bwd_time = benchmark_fn(lambda: ref_loss.backward())

    # # 打印性能对比结果
    # print(f"\n=== 性能对比 ({kl_type}) ===")
    # print(f"前向传播 | Triton: {tri_fwd_time:.4f} ms | PyTorch: {torch_fwd_time:.4f} ms | 加速比: {torch_fwd_time/tri_fwd_time:.2f}x")
    # # print(f"反向传播 | Triton: {tri_bwd_time:.4f} ms | PyTorch: {torch_bwd_time:.4f} ms | 加速比: {torch_bwd_time/tri_bwd_time:.2f}x")


    assert all_close("forward", tri_loss, ref_loss, rtol=1e-2, atol=1e-4)
    assert all_close("backward", log_probs.grad, std_log_probs.grad, rtol=1e-2, atol=1e-4)

    print('forward accuracy: ', torch.allclose(tri_loss, ref_loss, rtol=1e-2, atol=1e-4))
    print('backward accuracy: ', torch.allclose(log_probs.grad, std_log_probs.grad, rtol=1e-2, atol=1e-4))


    
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
            ms = triton.testing.do_bench(lambda: triton_grpo_loss(log_probs, log_probs1, log_probs2,
                        advantages, kl_type, kl_coef, loss_factor, clip, BLOCK_SIZE_T), warmup=warmup, rep=rep)
        if 'pytorch' in provider:
            ms = triton.testing.do_bench(lambda: torch_grpo_loss(std_log_probs, std_log_probs1, std_advantages,
                std_log_probs2, kl_type, kl_coef, loss_factor, clip), warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


class TestGRPOLoss:

    def test_grpo_loss(self):
        grpo_loss_kernel("kl")
        grpo_loss_kernel("unbias")
        grpo_loss_kernel("mse")
