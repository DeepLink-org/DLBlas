# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.
"""
import torch
import triton

from dlblas.kernels.fused_moe_v2 import fused_moe
from dlblas.layers.moe.kernels.blocked_fp8_fused_moe import dlblas_fused_moe_blocked_fp8
from dlblas.utils.device_utils import infer_device


def _make_A(M, K, group_size, out_dtype, device):
    quant_A = torch.rand(M, K // group_size, group_size, dtype=torch.float32, device=device)
    # -1 ~ 1
    quant_A = quant_A * 2 - 1
    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
    quant_A *= scaling
    quant_A = quant_A.to(out_dtype).to(torch.float32)

    # create scale and A
    scale = torch.rand(M, K // group_size, dtype=torch.float32, device=device)
    scale /= fmax
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    return A.to(torch.bfloat16), quant_A, scale


def _make_B(E, N, K, group_size, out_dtype, device):
    quant_B = torch.rand(E,
                         N // group_size,
                         group_size,
                         K // group_size,
                         group_size,
                         dtype=torch.float32,
                         device=device)
    quant_B = quant_B * 2 - 1

    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_B.abs().amax((2, 4), keepdim=True)
    quant_B *= scaling
    quant_B = quant_B.to(out_dtype).to(torch.float32)

    scale = torch.rand(E, N // group_size, 1, K // group_size, 1, dtype=torch.float32, device=device)
    scale /= fmax

    B = quant_B * scale

    B = B.reshape(E, N, K)
    quant_B = quant_B.reshape(E, N, K).to(out_dtype)
    scale = scale.reshape(E, N // group_size, K // group_size)
    return B.to(torch.bfloat16), quant_B, scale


def test_fused_moe():
    m, n, k = 80 * 1024, 128, 256
    e, topk, ep_size = 256, 8, 2
    chunk_size = 8 * 1024
    dtype = torch.bfloat16
    group_size = 128
    quant_dtype = torch.float8_e4m3fn
    local_e = e // ep_size
    device = infer_device()
    a, a_quant, a_scale = _make_A(m, k, group_size=group_size, out_dtype=quant_dtype, device=device)
    w1, w1_quant, w1_scale = _make_B(local_e, 2 * n, k, group_size=group_size, out_dtype=quant_dtype, device=device)
    w2, w2_quant, w2_scale = _make_B(local_e, k, n, group_size=group_size, out_dtype=quant_dtype, device=device)
    score = torch.randn((m, e), device=device, dtype=dtype)
    e_map = torch.arange(e, device=device, dtype=torch.int32)
    e_map[e_map >= local_e] = -1
    score = torch.rand(m, e, dtype=dtype, device=device)
    routing_weights = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_idx = torch.topk(routing_weights, topk, dim=-1)

    topk_weight[topk_idx < local_e] = 0.0
    topk_idx[topk_idx < local_e] = -1
    topk_idx[topk_idx >= local_e] -= local_e
    dlblas_topk_weight, dlblas_topk_idx = topk_weight.clone(), topk_idx.clone()

    triton_output = fused_moe(a,
                              w1_quant,
                              w2_quant,
                              topk_weight,
                              topk_idx,
                              inplace=True,
                              global_num_experts=e,
                              num_local_experts=local_e,
                              expert_map=e_map,
                              use_fp8_w8a8=True,
                              w1_scale=w1_scale,
                              w2_scale=w2_scale,
                              block_shape=[group_size, group_size],
                              chunk_size=chunk_size)
    dlblas_output = dlblas_fused_moe_blocked_fp8(a_quant,
                                                 a_scale,
                                                 w1_quant,
                                                 w1_scale,
                                                 w2_quant,
                                                 w2_scale,
                                                 topk_weights=dlblas_topk_weight,
                                                 topk_ids=dlblas_topk_idx,
                                                 topk=topk,
                                                 renormalize=False,
                                                 out_dtype=dtype,
                                                 expert_offset=local_e,
                                                 ep_size=ep_size)
    print(f"vllm_out: {triton_output}")
    print(f"dlblas_out: {dlblas_output}")
    print(f"最大差值：{torch.max(torch.abs(triton_output - dlblas_output)).item()}")
    print(f"chunk_size={chunk_size}")
    torch.testing.assert_close(triton_output, dlblas_output, atol=0.09, rtol=0)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_length'],
            x_vals=[m],
            line_arg='provider',
            line_vals=['v2', 'base'],
            line_names=['v2', 'base'],
            styles=[('blue', '-'), ('red', '-')],
            ylabel='us',
            plot_name='fused-moe-performance',
            args={},
        ))
    def benchmark(seq_length, provider):
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'v2':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_moe(a,
                                  w1_quant,
                                  w2_quant,
                                  topk_weight,
                                  topk_idx,
                                  inplace=True,
                                  global_num_experts=e,
                                  expert_map=e_map,
                                  use_fp8_w8a8=True,
                                  w1_scale=w1_scale,
                                  w2_scale=w2_scale,
                                  block_shape=[group_size, group_size],
                                  chunk_size=chunk_size),
                quantiles=quantiles,
            )
        elif provider == 'base':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: dlblas_fused_moe_blocked_fp8(a_quant,
                                                     a_scale,
                                                     w1_quant,
                                                     w1_scale,
                                                     w2_quant,
                                                     w2_scale,
                                                     topk_weights=dlblas_topk_weight,
                                                     topk_ids=dlblas_topk_idx,
                                                     topk=topk,
                                                     renormalize=False,
                                                     ep_size=e),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    benchmark.run(print_data=True)


if __name__ == '__main__':
    test_fused_moe()
