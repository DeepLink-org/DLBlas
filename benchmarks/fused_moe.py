# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import functools
import pytest
import torch
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import vllm.model_executor.layers.fused_moe  # noqa
from tests.kernels.utils import (opcheck, stack_and_dev, torch_moe,
                                 torch_moe_single)
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize, marlin_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


import triton
import triton.language as tl
from packaging import version

TRITON_VERSION = version.parse(triton.__version__)

if TRITON_VERSION >= version.parse('3.0.0'):
    fast_expf = tl.math.exp
else:
    fast_expf = tl.math.fast_expf


@triton.jit
def _silu_and_mul_kernel(
    gateup_ptr,
    out_ptr,
    N: tl.constexpr,
    M,
    stride_gum: tl.constexpr,
    stride_gun: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """silu and mul kernel."""
    n_block_id = tl.program_id(0)
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    up_ptr = gateup_ptr + N * stride_gun
    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if N % BLOCK_SIZE_N == 0:
        mask = None
    else:
        mask = offs_n < N

    gate_ptrs = gateup_ptr + m_id_start * stride_gum + offs_n * stride_gun
    up_ptrs = up_ptr + m_id_start * stride_gum + offs_n * stride_gun
    out_ptrs = out_ptr + m_id_start * stride_om + offs_n * stride_on

    for _ in tl.range(m_id_start, M, m_id_stride):
        gate = tl.load(gate_ptrs, mask=mask)
        up = tl.load(up_ptrs, mask=mask)
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)

        gate = gate / (1 + fast_expf(-gate))
        out = gate * up

        tl.store(out_ptrs, out, mask=mask)

        gate_ptrs += m_id_stride * stride_gum
        up_ptrs += m_id_stride * stride_gum
        out_ptrs += m_id_stride * stride_om


def silu_and_mul(gate_up: torch.Tensor, out: torch.Tensor = None):
    """silu and mul."""
    assert gate_up.dim() == 2

    M = gate_up.size(0)
    N = gate_up.size(-1) // 2
    if out is None:
        out_shape = (M, N)
        out = gate_up.new_empty(out_shape)

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, 512)
    num_warps = 4
    num_stages = 1

    props = get_device_props(gate_up.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    grid_size0 = triton.cdiv(N, BLOCK_SIZE_N)
    grid_size1 = min(M, num_sm * warps_per_sm // num_warps)
    assert grid_size0 < 65536 and grid_size1 < 65536
    grid = (grid_size0, grid_size1)
    _silu_and_mul_kernel[grid](gate_up,
                               out,
                               N,
                               M,
                               stride_gum=gate_up.stride(0),
                               stride_gun=gate_up.stride(1),
                               stride_om=out.stride(0),
                               stride_on=out.stride(1),
                               BLOCK_SIZE_N=BLOCK_SIZE_N,
                               num_warps=num_warps,
                               num_stages=num_stages)

    return out

if version.parse(triton.__version__) <= version.parse('2.2.0'):

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        from triton.runtime.jit import get_cuda_stream

        device = tensor.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)
else:

    KERNEL_META = dict()

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        return KERNEL_META


@triton.jit
def _start_end_kernel(TopkIdx, SortedIdx, ExpStart, ExpEnd, len_sorted_idx: int, num_experts: tl.constexpr,
                      BLOCK: tl.constexpr):
    """start end kernel."""
    exp_id = tl.program_id(0)
    exp_start = -1
    cnt = 0

    s_off = tl.arange(0, BLOCK)

    # find start
    for sidx_start in range(0, len_sorted_idx, BLOCK):
        sidx_off = sidx_start + s_off
        sidx_mask = sidx_off < len_sorted_idx
        sidx = tl.load(SortedIdx + sidx_off, mask=sidx_mask, other=0)
        tidx = tl.load(TopkIdx + sidx, mask=sidx_mask, other=num_experts)
        tidx_mask = tidx == exp_id
        cnt += tl.sum(tidx_mask.to(tl.int32))
        if cnt > 0 and exp_start < 0:
            exp_start = sidx_start + tl.argmax(tidx_mask, axis=0)

    if exp_start < 0:
        exp_start *= 0
    exp_end = exp_start + cnt
    tl.store(ExpStart + exp_id, exp_start)
    tl.store(ExpEnd + exp_id, exp_end)


def dlblas_get_start_end(topk_idx: torch.Tensor, sorted_idx: torch.Tensor, num_experts: int):
    """get start and end.
    same process as:
    >>> exp_tok_cnt = F.one_hot(flatten_topk_ids, num_classes=E).sum(0)
    >>> exp_end = exp_tok_cnt.cumsum(0)
    >>> exp_start = exp_end - exp_tok_cnt
    """
    start_end = sorted_idx.new_empty(2, num_experts)
    exp_start = start_end[0, :]
    exp_end = start_end[1, :]

    BLOCK = 128
    kernel_meta = get_kernel_meta(topk_idx)
    _start_end_kernel[(num_experts, )](
        topk_idx,
        sorted_idx,
        exp_start,
        exp_end,
        len_sorted_idx=sorted_idx.numel(),
        num_experts=num_experts,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
        **kernel_meta,
    )

    return exp_start, exp_end


def _dlblas_get_sorted_idx(topk_ids: torch.Tensor, num_experts: int):
    """get sorted idx."""
    flatten_topk_ids = topk_ids.flatten()
    sorted_idx = flatten_topk_ids.argsort()
    exp_start, exp_end = dlblas_get_start_end(flatten_topk_ids, sorted_idx, num_experts)
    return sorted_idx, exp_start, exp_end


def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
        }, num_stages=4, num_warps=4),
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K', 'M_NP2'],
    warmup=10,
    rep=25,
)
@triton.jit
def fused_moe_blocked_f8_kernel(
    A,
    A_scale,
    B,
    B_scale,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    Weights,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm,
    stride_ask: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_NP2: tl.constexpr,
    ENABLE_WEIGHTS: tl.constexpr,
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """fused moe kernel."""
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id + expert_offset)
    exp_end = tl.load(ExpEnd + exp_id + expert_offset)
    M = exp_end - exp_start
    if M <= 0:
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_sid = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_sid, other=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if reindex_a:
        offs_am = sid // top_k
    else:
        offs_am = offs_sid
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # deepseek has 160 experts, exp index would overflow int32
    exp_id = exp_id.to(tl.int64)
    exp_off = stride_be * exp_id
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = pid_n * BLOCK_SIZE_N // group_bn
    as_ptrs = A_scale + offs_am * stride_asm
    bs_ptrs = B_scale + stride_bse * exp_id + offs_bsn * stride_bsn

    acc_scale = tl.load(as_ptrs, mask=mask_sid, other=1.0) * tl.load(bs_ptrs)
    acc_ratio = 1 / acc_scale
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # load scales
        k_start = (k + 1) * BLOCK_SIZE_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=mask_sid and k_start < K, other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # mma
        accumulator = tl.dot(a, b, acc=accumulator * acc_ratio[:, None])

        # update scales and ratio
        new_acc_scale = a_scale * b_scale
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator * (acc_ratio * acc_scale)[:, None]

    if ENABLE_WEIGHTS:
        weight = tl.load(Weights + sid, mask=mask_sid)
        c = c * weight[:, None].to(c.dtype)

    c = c.to(C.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])



@triton.jit
def _quant_fp8_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    M,
    M_out,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sm,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
):
    """quant fp8 kernel."""
    group_id = tl.program_id(0)
    m_id_start = tl.program_id(1)
    m_id_stride = tl.num_programs(1)

    g_offs = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    g_offs = tl.max_contiguous(tl.multiple_of(g_offs, GROUP_SIZE), GROUP_SIZE)
    rfp8_max = 1 / fp8_max

    m_id = m_id_start
    a_ptrs = a_ptr + m_id * stride_am + g_offs * stride_ak
    o_ptrs = out_ptr + m_id * stride_om + g_offs * stride_ok
    s_ptr = scale_ptr + m_id * stride_sm + group_id * stride_sg

    for m_id in tl.range(m_id_start, M_out, m_id_stride):

        a = tl.load(a_ptrs, mask=m_id < M, other=0).to(tl.float32)
        scale = tl.max(tl.abs(a)) * rfp8_max
        out = a / scale

        out = tl.clamp(out, fp8_min, fp8_max)
        out = out.to(out_ptr.dtype.element_ty)

        tl.store(o_ptrs, out)
        tl.store(s_ptr, scale)

        a_ptrs += m_id_stride * stride_am
        o_ptrs += m_id_stride * stride_om
        s_ptr += m_id_stride * stride_sm

WARPS_PER_SM = {
    (8, 0): 64,
    (8, 6): 48,
    (8, 7): 48,
    (8, 9): 48,
    (9, 0): 64,
    (10, 0): 64,
    (10, 1): 48,
    (12, 0): 48,
}

@functools.lru_cache
def get_device_props(device=None):
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    warps_per_sm = WARPS_PER_SM.get((props.major, props.minor), 32)
    out = dict(
        multi_processor_count=props.multi_processor_count,
        warps_per_sm=warps_per_sm,
    )
    return out

def _quant_fp8_launcher(A: torch.Tensor, group_size: int, out: torch.Tensor, scales: torch.Tensor):
    """quant online."""
    M, K = A.shape
    num_groups = K // group_size
    M_out = out.size(0)

    dtype = out.dtype
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    num_warps = 1
    num_stages = 1

    props = get_device_props(A.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    max_ctas = num_sm * warps_per_sm // num_warps
    grid_size1 = min(M_out, max_ctas // num_groups)
    assert grid_size1 < 65536
    grid = (num_groups, grid_size1)
    _quant_fp8_kernel[grid](
        A,
        out,
        scales,
        M,
        M_out,
        fp8_min=fmin,
        fp8_max=fmax,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        stride_sm=scales.stride(0),
        stride_sg=scales.stride(1),
        GROUP_SIZE=group_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out, scales


def quant_fp8(A: torch.Tensor, group_size: int, dtype: torch.dtype = torch.float8_e4m3fn):
    """quant fp8."""
    assert A.dim() == 2
    M, K = A.shape
    assert K % group_size == 0
    num_groups = K // group_size
    out = torch.empty_like(A, dtype=dtype)
    scales = A.new_empty(M, num_groups, dtype=torch.float32)
    return _quant_fp8_launcher(A, group_size, out, scales)


def fused_moe_blocked_fp8_kernel_launcher(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    weights: torch.Tensor,
    enable_weights: bool = False,
    top_k: int = 1,
    num_tokens: int = None,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
):
    """fused moe kernel launcher."""

    if num_tokens is None:
        num_tokens = A.size(0)
    M_NP2 = triton.next_power_of_2(num_tokens)
    M_NP2 = max(64, M_NP2)
    E, N, K = B.shape

    assert A.dim() == 2
    assert A_scale.dim() == 2
    assert B.dim() == 3
    assert B_scale.dim() == 3

    assert K % A_scale.size(1) == 0
    assert K % B_scale.size(2) == 0
    assert N % B_scale.size(1) == 0

    group_ak = K // A_scale.size(1)
    group_bk = K // B_scale.size(2)
    group_bn = N // B_scale.size(1)

    def _grid_fn(META):
        grid = (triton.cdiv(M_NP2, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), E)
        return grid

    A = A.flatten(0, -2)
    C = C.flatten(0, -2)

    BLOCK_SIZE_K = group_bk
    GROUP_SIZE_M = 8
    grid = _grid_fn
    fused_moe_blocked_f8_kernel[grid](
        A,
        A_scale,
        B,
        B_scale,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        weights,
        N=N,
        K=K,
        group_ak=group_ak,
        group_bk=group_bk,
        group_bn=group_bn,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_asm=A_scale.stride(0),
        stride_ask=A_scale.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_bse=B_scale.stride(0),
        stride_bsn=B_scale.stride(1),
        stride_bsk=B_scale.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        ENABLE_WEIGHTS=enable_weights,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        M_NP2=M_NP2,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )


def _renormalize(topk_weights: torch.Tensor, renormalize: bool):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    return topk_weights
def _make_intermediate(shape: tuple, dtype: torch.dtype, device: torch.device, zeros: bool):
    """make intermediate."""
    if zeros:
        return torch.zeros(shape, dtype=dtype, device=device)
    else:
        return torch.empty(shape, dtype=dtype, device=device)

def dlblas_fused_moe_blocked_fp8(input: torch.Tensor,
                                 input_scale: torch.Tensor,
                                 w1: torch.Tensor,
                                 w1_scale: torch.Tensor,
                                 w2: torch.Tensor,
                                 w2_scale: torch.Tensor,
                                 topk_weights: torch.Tensor,
                                 topk_ids: torch.Tensor,
                                 topk: int,
                                 out_dtype: torch.dtype = torch.float16,
                                 expert_offset: int = 0,
                                 num_experts: int = None,
                                 renormalize: bool = False,
                                 ep_size: int = 1) -> torch.Tensor:
    """fused moe."""
    device = input.device
    M = input.size(0)
    E, N, _ = w1.shape
    if num_experts is None:
        num_experts = E
        if ep_size > 1:
            num_experts = num_experts * ep_size
    full_exp = num_experts == E
    group_size = input.size(-1) // input_scale.size(-1)

    topk_weights = _renormalize(topk_weights, renormalize)
    # dlblas get topk_ids
    topk_ids[topk_ids != -1] += expert_offset
    sorted_idx, exp_start, exp_end = _dlblas_get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N), dtype=out_dtype, device=device, zeros=not full_exp)
    # gate and up
    fused_moe_blocked_fp8_kernel_launcher(
        input,
        input_scale,
        w1,
        w1_scale,
        intermediate_cache1,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=False,
        top_k=topk,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=True,
        reindex_c=False,
    )

    # activate
    intermediate_cache1 = intermediate_cache1.flatten(0, -2)
    gate_cache = silu_and_mul(intermediate_cache1)
    del intermediate_cache1
    gate_cache, gate_scale = quant_fp8(gate_cache, group_size, dtype=input.dtype)

    intermediate_cache2 = _make_intermediate((M, topk, w2.shape[1]), dtype=out_dtype, device=device, zeros=not full_exp)
    # down
    fused_moe_blocked_fp8_kernel_launcher(
        gate_cache,
        gate_scale,
        w2,
        w2_scale,
        intermediate_cache2,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=True,
        top_k=1,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=False,
        reindex_c=True,
    )

    ret = intermediate_cache2.sum(dim=1)
    return ret


def _make_A(M, K, group_size, out_dtype, device='cuda'):
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


def _make_B(E, N, K, group_size, out_dtype, device='cuda'):
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

@pytest.mark.parametrize("m", [128])
# @pytest.mark.parametrize("n", [2048])
# @pytest.mark.parametrize("k", [7168])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("padding", [False])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    padding: bool,
):
    group_size = 128
    quant_dtype = torch.float8_e4m3fn
    local_e = e // ep_size
    a, a_quant, a_scale = _make_A(m, k, group_size=group_size, out_dtype=quant_dtype)
    w1, w1_quant, w1_scale = _make_B(local_e, 2*n, k, group_size=group_size, out_dtype=quant_dtype)
    w2, w2_quant, w2_scale = _make_B(local_e, k, n, group_size=group_size, out_dtype=quant_dtype)
    # print(f"a.shape={a.shape}, w1.shape={w1.shape}, w2.shape={w2.shape}")
    # a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    # w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    # w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    # print(f"a.shape={a.shape}, w1.shape={w1.shape}, w2.shape={w2.shape}")
    # quit()

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    
    # if ep_size > 1:
    # e_ids = torch.randint(0,
    #                         e, (local_e, ),
    #                         device="cuda",
    #                         dtype=torch.int32)
    
    # e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
    # e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    # w1 = w1[e_ids]
    # w2 = w2[e_ids]
    # else:
    #     e_map = None
    e_map = torch.arange(e, device="cuda", dtype=torch.int32)
    e_map[e_map >= local_e] = -1
    score = torch.rand(m, e, dtype=dtype, device="cuda")
    routing_weights = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, base_topk_idx = torch.topk(routing_weights, topk, dim=-1)
    topk_idx = base_topk_idx.clone()
    dlblas_topk_weight, dlblas_topk_idx = topk_weight.clone(), topk_idx.clone()
    dlblas_topk_idx[topk_idx < local_e] = -1
    dlblas_topk_idx[dlblas_topk_idx > 0] -= local_e
    dlblas_topk_weight[topk_idx < local_e] = 0.0

    # topk_idx[topk_idx>=local_e] = local_e + 1 #-1
    topk_idx[base_topk_idx < local_e] = -1
    topk_idx[base_topk_idx >= local_e] -= local_e
    topk_weight[base_topk_idx < local_e] = 0.0
    # print(topk_idx)
    # quit()


    torch_output = torch_moe(a, w1, w2, score, topk, e_map)
    # iterative_output = iterative_moe(a,
    #                                  w1,
    #                                  w2,
    #                                  score,
    #                                  topk,
    #                                  global_num_experts=e,
    #                                  expert_map=e_map,
    #                                  renormalize=False)

    # triton_output = fused_moe(a,
    #                           w1,
    #                           w2,
    #                         #   score,
    #                           topk_weight,
    #                           topk_idx,
    #                           topk,
    #                           global_num_experts=e,
    #                           expert_map=e_map,
    #                           renormalize=False)
    triton_output = fused_moe(a,
                              w1_quant,
                              w2_quant,
                              topk_weight,
                              topk_idx,
                              topk,
                              global_num_experts=e,
                              expert_map=e_map,
                              ep_rank=0,
                              use_fp8_w8a8=True,
                              w1_scale=w1_scale,
                              w2_scale=w2_scale,
                              block_shape=[group_size, group_size],
                              renormalize=False)
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
                                              out_dtype = torch.bfloat16,
                                              expert_offset = local_e,
                                              ep_size=ep_size)
    print(f"e_map={e_map}")
    # print(f"e_ids:{e_ids}")
    # print(f"torch_out:{torch_output}")
    print(f"vllm_out:{triton_output}")
    print(f"dlblas_out:{dlblas_output}")
    print(f"最大差值：{torch.max(torch.abs(triton_output - dlblas_output)).item()}")

    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[m],
        line_arg="provider",
        line_vals=["torch", "kernel"],
        line_names=["vllm", "dlblas Kernel"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="fused-moe-performance",
        args={},
    )
    )
    def benchmark(seq_length, provider):
        dtype = torch.bfloat16
        device = torch.device("cuda")
        num_experts, num_expert_group, topk_group, topk = 256, 8, 4, 8

        scores = torch.randn((seq_length, num_experts), device=device, dtype=dtype)
        bias = torch.rand(num_experts, device=device, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_moe(a,
                              w1_quant,
                              w2_quant,
                              topk_weight,
                              topk_idx,
                              topk,
                              global_num_experts=e,
                              expert_map=e_map,
                              use_fp8_w8a8=True,
                              w1_scale=w1_scale,
                              w2_scale=w2_scale,
                              block_shape=[group_size, group_size],
                              renormalize=False),
                quantiles=quantiles,
            )
        elif provider == "kernel":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: dlblas_fused_moe_blocked_fp8(a_quant,
                                              a_scale,
                                              w1_quant,
                                              w1_scale,
                                              w2_quant,
                                              w2_scale,
                                              topk_weights=topk_weight,
                                              topk_ids=topk_idx,
                                              topk=topk,
                                              renormalize=False,
                                              ep_size=e),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms
    benchmark.run(print_data=True)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    torch.testing.assert_close(triton_output, dlblas_output, atol=2e-2, rtol=0)
    assert False
    # torch.testing.assert_close(iterative_output,
    #                            torch_output,
    #                            atol=2e-2,
    #                            rtol=0)

'''
@pytest.mark.parametrize("m", [1, 32, 222])
@pytest.mark.parametrize("n", [128, 1024, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("has_zp", [True, False])
@pytest.mark.parametrize("weight_bits", [4, 8])
def test_fused_moe_wn16(m: int, n: int, k: int, e: int, topk: int,
                        ep_size: int, dtype: torch.dtype, group_size: int,
                        has_zp: bool, weight_bits: int):
    print(m, n, k, e, topk, dtype, group_size, has_zp, weight_bits)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qweight = torch.empty((e, 2 * n, k // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor),
                             device="cuda",
                             dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size),
                            device="cuda",
                            dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size),
                            device="cuda",
                            dtype=dtype)
    w1_qzeros = torch.empty((e, 2 * n // pack_factor, k // group_size),
                            device="cuda",
                            dtype=torch.uint8)
    w2_qzeros = torch.empty((e, k // pack_factor, n // group_size),
                            device="cuda",
                            dtype=torch.uint8)

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w1, w1_ref, w1_qweight, w1_scales, w1_qzeros
        else:
            w, w_ref, w_qweight, w_scales, w_qzeros = \
                w2, w2_ref, w2_qweight, w2_scales, w2_qzeros
        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False)
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            if has_zp:
                qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        w_ref[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randint(0,
                              e, (local_e, ),
                              device="cuda",
                              dtype=torch.int32)
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1_ref = w1_ref[e_ids]
        w2_ref = w2_ref[e_ids]
        w1_qweight = w1_qweight[e_ids]
        w2_qweight = w2_qweight[e_ids]
        w1_scales = w1_scales[e_ids]
        w2_scales = w2_scales[e_ids]
        w1_qzeros = w1_qzeros[e_ids]
        w2_qzeros = w2_qzeros[e_ids]
    else:
        e_map = None

    triton_output = fused_moe(a,
                              w1_qweight,
                              w2_qweight,
                              score,
                              topk,
                              renormalize=False,
                              use_int4_w4a16=weight_bits == 4,
                              use_int8_w8a16=weight_bits == 8,
                              global_num_experts=e,
                              expert_map=e_map,
                              w1_scale=w1_scales,
                              w2_scale=w2_scales,
                              w1_zp=w1_qzeros if has_zp else None,
                              w2_zp=w2_qzeros if has_zp else None,
                              block_shape=[0, group_size])
    torch_output = torch_moe(a, w1_ref, w2_ref, score, topk, e_map)
    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False])
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype, padding: bool, use_rocm_aiter: bool,
                     monkeypatch):
    """Make sure our Mixtral MoE implementation agrees with the one from
    huggingface."""

    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        params_dtype=dtype,
        tp_size=1,
        dp_size=1,
    ).cuda()

    # Load the weights
    vllm_moe.gate.weight.data[:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.experts.w13_weight[i][:] = torch.cat(weights, dim=0)
        vllm_moe.experts.w2_weight[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    hf_inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")
    # vLLM uses 1D query [num_tokens, hidden_dim]
    vllm_inputs = hf_inputs.flatten(0, 1)

    # Pad the weight if moe padding is enabled
    if padding:
        vllm_moe.experts.w13_weight = Parameter(F.pad(
            vllm_moe.experts.w13_weight, (0, 128), "constant", 0)[..., 0:-128],
                                                requires_grad=False)
        torch.cuda.empty_cache()
        vllm_moe.experts.w2_weight = Parameter(F.pad(
            vllm_moe.experts.w2_weight, (0, 128), "constant", 0)[..., 0:-128],
                                               requires_grad=False)
        torch.cuda.empty_cache()

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(hf_inputs)
    vllm_states = vllm_moe.forward(vllm_inputs)

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    if use_rocm_aiter:
        # The values of rtol and atol are set based on the tests in ROCM AITER package. # noqa: E501
        # https://github.com/ROCm/aiter/blob/dfed377f4be7da96ca2d75ac0761f569676f7240/op_tests/test_moe.py#L174  # noqa: E501
        torch.testing.assert_close(hf_states.flatten(0, 1),
                                   vllm_states,
                                   rtol=0.01,
                                   atol=100)
    else:
        torch.testing.assert_close(hf_states.flatten(0, 1),
                                   vllm_states,
                                   rtol=mixtral_moe_tol[dtype],
                                   atol=mixtral_moe_tol[dtype])


@pytest.mark.parametrize("m", [1, 33, 123])
@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("k", [256, 2048])
@pytest.mark.parametrize("e", [4, 12])
@pytest.mark.parametrize("topk", [2, 3])
@pytest.mark.parametrize("ep_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("has_zp", [True, False])
@pytest.mark.parametrize("is_k_full", [True, False])
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
def test_fused_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    group_size: int,
    act_order: bool,
    num_bits: int,
    has_zp: bool,
    is_k_full: bool,
):
    current_platform.seed_everything(7)

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size in (k, n):
            return
        if has_zp:
            return
    else:
        if not is_k_full:
            return

    if has_zp:
        # we don't build kernel for int8 with zero
        if num_bits == 8:
            return
        quant_type = scalar_types.uint4 if num_bits == 4 else scalar_types.uint8
    else:
        quant_type = scalar_types.uint4b8 \
                if num_bits == 4 else scalar_types.uint8b128
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randperm(e, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    w_ref1_l = []
    qweight1_l = []
    scales1_l = []
    zeros1_l = []
    g_idx1_l = []
    sort_indices1_l = []

    for i in range(w1.shape[0]):
        if has_zp:
            w_ref1, qweight1, scales1, zeros1 = awq_marlin_quantize(
                w1[i].transpose(1, 0), quant_type, group_size)

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            zeros1_l.append(zeros1)
        else:
            test_perm = torch.randperm(k)
            quant_res = marlin_quantize(w1[i].transpose(1, 0), quant_type,
                                        group_size, act_order, test_perm)
            w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = quant_res

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            g_idx1_l.append(g_idx1)
            sort_indices1_l.append(sort_indices1)

    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    g_idx1 = stack_and_dev(g_idx1_l) if g_idx1_l else None
    zeros1 = stack_and_dev(zeros1_l) if zeros1_l else None
    sort_indices1 = stack_and_dev(sort_indices1_l) if sort_indices1_l else None

    w_ref2_l = []
    qweight2_l = []
    scales2_l = []
    zeros2_l = []
    g_idx2_l = []
    sort_indices2_l = []

    for i in range(w2.shape[0]):
        if has_zp:
            w_ref2, qweight2, scales2, zeros2 = awq_marlin_quantize(
                w2[i].transpose(1, 0), quant_type, group_size)

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            zeros2_l.append(zeros2)
        else:
            test_perm = torch.randperm(n)
            quant_res = marlin_quantize(w2[i].transpose(1, 0), quant_type,
                                        group_size, act_order, test_perm)
            w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = quant_res

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            g_idx2_l.append(g_idx2)
            sort_indices2_l.append(sort_indices2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    g_idx2 = stack_and_dev(g_idx2_l) if g_idx2_l else None
    zeros2 = stack_and_dev(zeros2_l) if zeros2_l else None
    sort_indices2 = stack_and_dev(sort_indices2_l) if sort_indices2_l else None

    score = torch.randn((m, e), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(a, score, topk, False)

    torch_output = torch_moe(a, w_ref1, w_ref2, score, topk, e_map)

    marlin_output = torch.ops.vllm.fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=e_map,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        num_bits=num_bits,
        is_k_full=is_k_full)

    torch.testing.assert_close(marlin_output, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skip("This test is here for the sake of debugging, "
                  "don't run it in automated tests.")
@pytest.mark.parametrize("m", [1, 33, 123])
@pytest.mark.parametrize("n", [128, 1024])
@pytest.mark.parametrize("k", [256, 2048])
@pytest.mark.parametrize("e", [4, 12])
@pytest.mark.parametrize("topk", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("has_zp", [True, False])
@pytest.mark.parametrize("is_k_full", [True, False])
def test_single_marlin_moe_multiply(m: int, n: int, k: int, e: int, topk: int,
                                    dtype: torch.dtype, group_size: int,
                                    act_order: bool, num_bits: int,
                                    has_zp: bool, is_k_full: bool):
    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size in (k, n):
            return
        if has_zp:
            return
    else:
        if not is_k_full:
            return

    if has_zp:
        quant_type = scalar_types.uint4 if num_bits == 4 else scalar_types.uint8
    else:
        quant_type = scalar_types.uint4b8 \
                if num_bits == 4 else scalar_types.uint8b128
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10

    w_ref_l = []
    qweight_l = []
    scales_l = []
    zeros_l = []
    g_idx_l = []
    sort_indices_l = []

    for i in range(w.shape[0]):
        if has_zp:
            w_ref, qweight, scales, zeros = awq_marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size)

            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            zeros_l.append(zeros)
        else:
            test_perm = torch.randperm(k)
            w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size, act_order,
                test_perm)

            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            g_idx_l.append(g_idx)
            sort_indices_l.append(sort_indices)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweight_l).contiguous()
    scales = stack_and_dev(scales_l)
    g_idx = stack_and_dev(g_idx_l) if g_idx_l else None
    zeros = stack_and_dev(zeros_l) if zeros_l else None
    sort_indices = stack_and_dev(sort_indices_l) if sort_indices_l else None

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    marlin_output = torch.ops.vllm.single_marlin_moe(
        a,
        qweight,
        scales,
        score,
        topk,
        renormalize=False,
        g_idx=g_idx,
        sort_indices=sort_indices,
        w_zeros=zeros,
        num_bits=num_bits,
        is_k_full=is_k_full,
    )

    torch_output = torch_moe_single(a, w_ref, score, topk)

    torch.testing.assert_close(marlin_output, torch_output, atol=2e-2, rtol=0)


def test_moe_align_block_size_opcheck():
    num_experts = 4
    block_size = 4
    topk_ids = torch.randint(0,
                             num_experts, (3, 4),
                             dtype=torch.int32,
                             device='cuda')

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    opcheck(torch.ops._moe_C.moe_align_block_size,
            (topk_ids, num_experts, block_size, sorted_ids, expert_ids,
             num_tokens_post_pad))
'''