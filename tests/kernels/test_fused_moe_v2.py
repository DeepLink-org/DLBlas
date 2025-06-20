# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.
"""
import pytest
import torch

from dlblas.kernels.fused_moe_v2 import fused_moe
from dlblas.layers.moe.kernels.blocked_fp8_fused_moe import dlblas_fused_moe_blocked_fp8
from dlblas.utils.device_utils import infer_device

DEVICE = infer_device()


def _make_A(M, K, group_size, out_dtype, device=DEVICE):
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


def _make_B(E, N, K, group_size, out_dtype, device=DEVICE):
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


NUM_EXPERTS = [256]
EP_SIZE = [2]
TOP_KS = [8]


@pytest.mark.parametrize('m', [0, 80, 800, 8000, 80 * 1024])
# @pytest.mark.parametrize("n", [2048])
# @pytest.mark.parametrize("k", [7168])
@pytest.mark.parametrize('n', [128])
@pytest.mark.parametrize('k', [128])
@pytest.mark.parametrize('e', NUM_EXPERTS)
@pytest.mark.parametrize('topk', TOP_KS)
@pytest.mark.parametrize('ep_size', EP_SIZE)
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('quant', [False, True])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    inplace: bool,
    quant: bool,
):
    group_size = 128
    quant_dtype = torch.float8_e4m3fn
    local_e = e // ep_size
    a, a_quant, a_scale = _make_A(m, k, group_size=group_size, out_dtype=quant_dtype)
    w1, w1_quant, w1_scale = _make_B(local_e, 2 * n, k, group_size=group_size, out_dtype=quant_dtype)
    w2, w2_quant, w2_scale = _make_B(local_e, k, n, group_size=group_size, out_dtype=quant_dtype)

    score = torch.randn((m, e), device=DEVICE, dtype=dtype)
    e_map = torch.arange(e, device=DEVICE, dtype=torch.int32)
    e_map[e_map >= local_e] = -1
    score = torch.rand(m, e, dtype=dtype, device=DEVICE)
    routing_weights = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_idx = torch.topk(routing_weights, topk, dim=-1)
    topk_weight[topk_idx < local_e] = 0.0
    topk_idx[topk_idx < local_e] = -1
    topk_idx[topk_idx >= local_e] -= local_e

    triton_output = fused_moe(a_quant if quant else a,
                              w1_quant,
                              w2_quant,
                              topk_weight,
                              topk_idx,
                              inplace=False if quant else inplace,
                              global_num_experts=e,
                              num_local_experts=local_e,
                              expert_map=e_map,
                              use_fp8_w8a8=True,
                              w1_scale=w1_scale,
                              w2_scale=w2_scale,
                              hidden_states_scale=a_scale if quant else None,
                              block_shape=[group_size, group_size],
                              chunk_size=32 * 1024)
    dlblas_output = dlblas_fused_moe_blocked_fp8(a_quant,
                                                 a_scale,
                                                 w1_quant,
                                                 w1_scale,
                                                 w2_quant,
                                                 w2_scale,
                                                 topk_weights=topk_weight,
                                                 topk_ids=topk_idx,
                                                 topk=topk,
                                                 renormalize=False,
                                                 out_dtype=dtype,
                                                 expert_offset=local_e,
                                                 ep_size=ep_size)

    torch.testing.assert_close(triton_output, dlblas_output, atol=0.07, rtol=0)
