# Copyright (c) 2025, DeepLink.
import random
import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.fused_moe_v3 import ep_scatter, ep_gather


device_ = infer_device()
def ep_scatter_ref(block_size, num_recv_tokens_per_expert_list, output_indexs, recv_topk_id, all_tokens,
                   recv_x, recv_x_scale):
    m_indices_ref = torch.empty(all_tokens, device=device_, dtype=torch.int32)
    output_tensor_ref = torch.zeros((all_tokens, 4096), device=device_, dtype=torch.float32).to(torch.float16)
    output_tensor_scale_ref = torch.zeros((all_tokens, 4096 // block_size), device=device_, dtype=torch.float32)
    cur = 0
    for i, k in enumerate(num_recv_tokens_per_expert_list):
        m_indices_ref[cur : cur + k] = i
        cur += k
    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = output_indexs[i][j]
                output_tensor_ref[dst][:] = recv_x[i][:]
                output_tensor_scale_ref[dst][:] = recv_x_scale[i][:]
    return m_indices_ref, output_tensor_ref, output_tensor_scale_ref


def ep_scatter_wrap(block_size, num_recv_tokens_per_expert_list, output_indexs, recv_topk_id, all_tokens,
                recv_x, recv_x_scale):
    num_recv_tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, dtype=torch.int, device=device_)
    output_tensor = torch.zeros((all_tokens, 4096), device=device_, dtype=torch.float32).to(torch.float16)
    expert_start_loc = torch.cumsum(torch.tensor([0] + num_recv_tokens_per_expert_list[:-1], device=device_), dim=0).to(torch.int32)
    output_tensor_scale = torch.zeros((all_tokens, 4096 // block_size), device=device_, dtype=torch.float32)
    m_indices = torch.empty(all_tokens, device=device_, dtype=torch.int32)
    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk_id,
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_indexs,
    )
    return m_indices, output_tensor, output_tensor_scale


def bench_scatter():
    # scatter
    block_size = 128
    num_recv_tokens_per_expert_list = [0] * 32
    num_recv_tokens_per_expert_list[6] = 128
    num_recv_tokens_per_expert_list[7] = 128
    num_recv_tokens_per_expert_list[8] = 128
    recv_topk_id = torch.ones((7, 8), device=device_, dtype=torch.int32) * -1
    for i in range(7):
        idx = random.randint(0, 7)
        expert_id = random.randint(6, 8)
        recv_topk_id[i][idx] = expert_id
    output_indexs = torch.zeros_like(recv_topk_id)
    all_tokens = sum(num_recv_tokens_per_expert_list)
    recv_x = torch.randn((7, 4096), device=device_, dtype=torch.float32).to(torch.float16)
    recv_x_scale = torch.randn((7, 4096 // block_size), device=device_, dtype=torch.float32)
    m_indices, output_tensor, output_tensor_scale = ep_scatter_wrap(block_size, 
                                                                    num_recv_tokens_per_expert_list,
                                                                    output_indexs=output_indexs,
                                                                    recv_topk_id=recv_topk_id,
                                                                    all_tokens=all_tokens,
                                                                    recv_x=recv_x,
                                                                    recv_x_scale=recv_x_scale)
    m_indices_ref, output_tensor_ref, output_tensor_scale_ref = ep_scatter_ref(block_size, 
                                                                               num_recv_tokens_per_expert_list,
                                                                               output_indexs=output_indexs,
                                                                               recv_topk_id=recv_topk_id,
                                                                               all_tokens=all_tokens,
                                                                               recv_x=recv_x,
                                                                               recv_x_scale=recv_x_scale)
    assert torch.allclose(m_indices, m_indices_ref, atol=1e-2, rtol=0)
    diff = output_tensor.to(torch.float) - output_tensor_ref.to(torch.float)
    print("june max =", torch.max(diff))
    assert torch.allclose(output_tensor.to(torch.float), output_tensor_ref.to(torch.float), atol=1e-2, rtol=0)
    assert torch.allclose(output_tensor_scale, output_tensor_scale_ref, atol=1e-2, rtol=0)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            ylabel='ms',
            plot_name='ep_scatter',
            args={},
        ))
    @triton.testing.perf_report(configs)
    def bench_fn(op, provider):
        warmup = 100
        rep = 200
        if 'torch' in provider:
            ms = triton.testing.do_bench(lambda: ep_scatter_ref(block_size, 
                                                                        num_recv_tokens_per_expert_list,
                                                                        output_indexs=output_indexs,
                                                                        recv_topk_id=recv_topk_id,
                                                                        all_tokens=all_tokens,
                                                                        recv_x=recv_x,
                                                                        recv_x_scale=recv_x_scale), 
                                            warmup=warmup, rep=rep)
        if 'triton' in provider:
            ms = triton.testing.do_bench(lambda: ep_scatter_wrap(block_size, 
                                                                    num_recv_tokens_per_expert_list,
                                                                    output_indexs=output_indexs,
                                                                    recv_topk_id=recv_topk_id,
                                                                    all_tokens=all_tokens,
                                                                    recv_x=recv_x,
                                                                    recv_x_scale=recv_x_scale), 
                                            warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=False, print_data=True)


def gather_ref(recv_x, gather_input, recv_topk_id, recv_topk_weights):
    output_indexs = torch.zeros_like(recv_topk_id)
    gather_out_ref = torch.zeros_like(recv_x, device=device_, dtype=torch.bfloat16)
    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = output_indexs[i][j]
                gather_out_ref[i][:] += gather_input[dst][:] * recv_topk_weights[i][j]
    return gather_out_ref


def gather_wrap(recv_x, gather_input, recv_topk_id, recv_topk_weights):
    output_indexs = torch.zeros_like(recv_topk_id)
    gather_out = torch.empty_like(recv_x, device=device_, dtype=torch.bfloat16)
    ep_gather(gather_input, recv_topk_id, recv_topk_weights, output_indexs, gather_out)
    return gather_out


def bench_gather():
    #### gather
    recv_x = torch.randn((7, 4096), device=device_, dtype=torch.float32).to(torch.float16)
    num_recv_tokens_per_expert_list = [0] * 32
    num_recv_tokens_per_expert_list[6] = 128
    num_recv_tokens_per_expert_list[7] = 128
    num_recv_tokens_per_expert_list[8] = 128
    all_tokens = sum(num_recv_tokens_per_expert_list)
    recv_topk_id = torch.ones((7, 8), device=device_, dtype=torch.int32) * -1
    recv_topk_weights = torch.zeros((7, 8), device=device_, dtype=torch.float)
    for i in range(7):
        idx = random.randint(0, 7)
        expert_id = random.randint(6, 8)
        recv_topk_id[i][idx] = expert_id
        recv_topk_weights[i][idx] = random.randint(0, 10) / 10.0
    output_indexs = torch.zeros_like(recv_topk_id)
    gather_out_ref = torch.zeros_like(recv_x, device=device_, dtype=torch.bfloat16)
    gather_out = torch.empty_like(recv_x, device=device_, dtype=torch.bfloat16)
    gather_input = torch.zeros((all_tokens, 4096), device=device_, dtype=torch.bfloat16)
    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = output_indexs[i][j]
                gather_out_ref[i][:] += gather_input[dst][:] * recv_topk_weights[i][j]
    ep_gather(gather_input, recv_topk_id, recv_topk_weights, output_indexs, gather_out)
    assert torch.allclose(gather_out, gather_out_ref, atol=1e-2, rtol=0)
    gather_ref(recv_x, gather_input, recv_topk_id, recv_topk_weights)
    gather_wrap(recv_x, gather_input, recv_topk_id, recv_topk_weights)
    assert torch.allclose(gather_out, gather_out_ref, atol=1e-2, rtol=0)
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            ylabel='ms',
            plot_name='ep_gather',
            args={},
        ))
    @triton.testing.perf_report(configs)
    def bench_fn(op, provider):
        warmup = 100
        rep = 200
        if 'torch' in provider:
            ms = triton.testing.do_bench(lambda:gather_ref(recv_x, gather_input, recv_topk_id, recv_topk_weights), 
                                            warmup=warmup, rep=rep)
        if 'triton' in provider:
            ms = triton.testing.do_bench(lambda: gather_wrap(recv_x, gather_input, recv_topk_id, recv_topk_weights), 
                                            warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    bench_scatter()
    bench_gather()