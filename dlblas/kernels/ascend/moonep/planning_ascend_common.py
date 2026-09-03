"""Shared MoonEP-compatible planning ABI and multi-rank transport helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the NPU backend


BLOCK_SIZE_P2 = 2048

DEDUP_MODE_ZERO = "zero"
DEDUP_MODE_CURRENT = "current"
DEDUP_MODE_SRC = "src"
DEDUP_MODES = (
    DEDUP_MODE_ZERO,
    DEDUP_MODE_CURRENT,
    DEDUP_MODE_SRC,
)


def normalize_dedup_mode(dedup_mode: str) -> str:
    """Validate and return one public Planning dedup mode."""
    if dedup_mode not in DEDUP_MODES:
        raise ValueError(f"dedup_mode must be one of {DEDUP_MODES}, got {dedup_mode!r}")
    return dedup_mode


def _trace(ctx: dict, message: str) -> None:
    """Print rank-tagged transport diagnostics when explicitly enabled."""
    if os.environ.get("MOONEP_PLANNING_TRACE") == "1":
        print(f"[planning rank={ctx.get('rank', '?')}] {message}", flush=True)


def _align_up(value: int, alignment: int) -> int:
    """Round an integer up to the requested alignment."""
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    """Standalone equivalent of MoonEP's caller-owned planning outputs."""

    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor


@dataclass(frozen=True, slots=True)
class PlanningGlobalOutputs:
    """Owner-rank storage holding every rank's planning result."""

    dst: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor


def make_planning_context(
    S: int,
    K: int,
    E: int,
    R: int,
    *,
    H: int = 16,
    B: int | None = None,
    num_sms: int = 1,
    token_padding: int = 128,
    rank: int = 0,
    group=None,
) -> dict:
    """Build a standalone context with MoonEP-compatible meta offsets."""
    if B is None:
        B = E // R
    assert S > 0 and K > 0 and R > 0
    assert E > 0 and E % R == 0
    assert H > 0 and B > 0 and num_sms > 0 and token_padding > 0
    assert 0 <= rank < R

    epn = E // R
    N = S * K
    NvS_capacity = N
    NvS = NvS_capacity + (token_padding - 1) * 2 * epn
    num_vblocks = (N + BLOCK_SIZE_P2 - 1) // BLOCK_SIZE_P2

    # Mirror MoonEP's int32 meta layout even though standalone tests do not
    # allocate a VMM-backed meta buffer.
    weights_off = 0
    tpe_off = _align_up(NvS, 4)
    plan_off = _align_up(tpe_off + R * E, 4)
    planning_out_elems = 3 * E * R + R * (E + B) + 2 * R * (E + B) + R * B + 2 * R
    n4 = _align_up(N, 4)
    topk0_off = _align_up(plan_off + planning_out_elems, 4)
    order_off = topk0_off + n4
    order0_off = order_off + n4
    barrier_off = order0_off + n4
    src_info_off = barrier_off + 3
    meta_chunk_logical = src_info_off + NvS
    meta_chunk_padded = _align_up(meta_chunk_logical, 4)

    return {
        "rank": int(rank),
        "group": group,
        "R": int(R),
        "E": int(E),
        "S": int(S),
        "K": int(K),
        "H": int(H),
        "B": int(B),
        "N": int(N),
        "NvS": int(NvS),
        "NvS_capacity": int(NvS_capacity),
        "num_sms": int(num_sms),
        "num_vblocks": int(num_vblocks),
        "token_padding": int(token_padding),
        "planning_out_elems": int(planning_out_elems),
        "meta_chunk_padded": int(meta_chunk_padded),
        "WEIGHTS_OFF": int(weights_off),
        "TPE_OFF": int(tpe_off),
        "PLAN_OFF": int(plan_off),
        "TOPK0_OFF": int(topk0_off),
        "ORDER_OFF": int(order_off),
        "ORDER0_OFF": int(order0_off),
        "BARRIER_OFF": int(barrier_off),
        "SRC_INFO_OFF": int(src_info_off),
    }


def _empty_i32(length: int, device: torch.device) -> torch.Tensor:
    """Allocate a tail-padded int32 vector and return its logical slice."""
    return torch.empty(_align_up(length, 4), dtype=torch.int32, device=device)[:length]


def allocate_planning_outputs(ctx: dict) -> tuple[MoonEPCommPlan, torch.Tensor]:
    """Allocate outputs with the same call and return contract as MoonEP."""
    device = torch.device("npu", torch.npu.current_device())
    N = int(ctx["N"])
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    NvS = int(ctx["NvS"])
    K = int(ctx["K"])

    plan = MoonEPCommPlan(
        dst=_empty_i32(N, device),
        experts_to_copy=_empty_i32(R * B, device).view(R, B),
        zero_fill_ranges=_empty_i32((E + B) * 2, device).view(E + B, 2),
        remote_stats=_empty_i32(2, device),
        N=N,
        R=R,
        E=E,
        B=B,
        NvS=NvS,
        K=K,
        dup_groups=_empty_i32(NvS * 3, device).view(NvS, 3),
        dup_loffs=_empty_i32(NvS, device),
        dup_counts=_empty_i32(2, device),
    )
    cu_seqlens = _empty_i32(E + B, device)
    return plan, cu_seqlens


def validate_planning_call(
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan: MoonEPCommPlan,
) -> None:
    """Validate the official rank-local Planning call contract."""
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    N = int(ctx["N"])
    NvS = int(ctx["NvS"])
    assert isinstance(plan, MoonEPCommPlan)
    assert topk_experts_flat.dtype == torch.int32
    assert tokens_per_expert.dtype == torch.int32
    assert topk_experts_flat.is_contiguous()
    assert tokens_per_expert.is_contiguous()
    assert topk_experts_flat.numel() == N
    assert tuple(tokens_per_expert.shape) == (E,)
    assert tuple(cu_seqlens.shape) == (E + B,)
    assert tuple(plan.dst.shape) == (N,)
    assert tuple(plan.experts_to_copy.shape) == (R, B)
    assert tuple(plan.zero_fill_ranges.shape) == (E + B, 2)
    assert tuple(plan.remote_stats.shape) == (2,)
    assert tuple(plan.dup_groups.shape) == (NvS, 3)
    assert tuple(plan.dup_loffs.shape) == (NvS,)
    assert tuple(plan.dup_counts.shape) == (2,)
    tensors = (
        topk_experts_flat,
        tokens_per_expert,
        cu_seqlens,
        plan.dst,
        plan.experts_to_copy,
        plan.zero_fill_ranges,
        plan.remote_stats,
        plan.dup_groups,
        plan.dup_loffs,
        plan.dup_counts,
    )
    for tensor in tensors:
        assert tensor.dtype == torch.int32
        assert tensor.device.type == "npu"


def allocate_global_outputs(ctx: dict) -> PlanningGlobalOutputs:
    """Allocate fixed-shape owner results on every rank for HCCL broadcast."""
    device = torch.device("npu", torch.npu.current_device())
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    N = int(ctx["N"])
    NvS = int(ctx["NvS"])
    return PlanningGlobalOutputs(
        dst=torch.zeros((R, N), dtype=torch.int32, device=device),
        cu_seqlens=torch.zeros((R, E + B), dtype=torch.int32, device=device),
        experts_to_copy=torch.full((R, B), -1, dtype=torch.int32, device=device),
        zero_fill_ranges=torch.zeros((R, E + B, 2), dtype=torch.int32, device=device),
        remote_stats=torch.zeros((R, 2), dtype=torch.int32, device=device),
        dup_groups=torch.zeros((R, NvS, 3), dtype=torch.int32, device=device),
        dup_loffs=torch.zeros((R, NvS), dtype=torch.int32, device=device),
        dup_counts=torch.zeros((R, 2), dtype=torch.int32, device=device),
    )


def gather_rank_inputs(
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect every physical rank's local inputs with real HCCL collectives."""
    R = int(ctx["R"])
    group = ctx.get("group")
    if R == 1:
        return topk_experts_flat.view(1, -1), tokens_per_expert.view(1, -1)
    assert dist.is_initialized(), "R > 1 Planning requires an initialized process group"
    assert dist.get_world_size(group) == R
    gathered_topk = torch.empty(
        R * topk_experts_flat.numel(),
        dtype=topk_experts_flat.dtype,
        device=topk_experts_flat.device,
    )
    gathered_tpe = torch.empty(
        R * tokens_per_expert.numel(),
        dtype=tokens_per_expert.dtype,
        device=tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(gathered_topk, topk_experts_flat, group=group)
    dist.all_gather_into_tensor(gathered_tpe, tokens_per_expert, group=group)
    return gathered_topk.view(R, -1), gathered_tpe.view(R, -1)


def _copy_rank_outputs(
    ctx: dict,
    outputs: PlanningGlobalOutputs,
    cu_seqlens: torch.Tensor,
    plan: MoonEPCommPlan,
) -> None:
    """Copy the addressed rank's global result into official local buffers."""
    rank = int(ctx["rank"])
    plan.dst.copy_(outputs.dst[rank])
    cu_seqlens.copy_(outputs.cu_seqlens[rank])
    plan.experts_to_copy.copy_(outputs.experts_to_copy)
    plan.zero_fill_ranges.copy_(outputs.zero_fill_ranges[rank])
    plan.remote_stats.copy_(outputs.remote_stats[rank])
    plan.dup_groups.copy_(outputs.dup_groups[rank])
    plan.dup_loffs.copy_(outputs.dup_loffs[rank])
    plan.dup_counts.copy_(outputs.dup_counts[rank])


RankPlanningCore = Callable[
    [
        dict,
        torch.Tensor,
        torch.Tensor,
        PlanningGlobalOutputs,
        str,
        torch.Tensor | None,
    ],
    int,
]


def run_all_rank_planning(
    impl_name: str,
    rank_core: RankPlanningCore,
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan: MoonEPCommPlan,
    *,
    dedup_mode: str = DEDUP_MODE_CURRENT,
) -> torch.Tensor | None:
    """Gather rank-local inputs and run complete Planning on every rank."""
    dedup_mode = normalize_dedup_mode(dedup_mode)
    _trace(ctx, f"{impl_name}: validate")
    validate_planning_call(ctx, topk_experts_flat, tokens_per_expert, cu_seqlens, plan)
    _trace(ctx, f"{impl_name}: gather inputs begin")
    all_topk, all_tpe = gather_rank_inputs(ctx, topk_experts_flat, tokens_per_expert)
    _trace(ctx, f"{impl_name}: gather inputs done")
    outputs = allocate_global_outputs(ctx)
    src = (
        torch.full(
            (int(ctx["NvS"]),),
            -1,
            dtype=torch.int32,
            device=topk_experts_flat.device,
        )
        if dedup_mode == DEDUP_MODE_SRC
        else None
    )
    _trace(ctx, f"{impl_name}: rank core begin")
    launch_count = int(rank_core(ctx, all_topk, all_tpe, outputs, dedup_mode, src))
    assert launch_count > 0
    if dedup_mode == DEDUP_MODE_SRC:
        assert src is not None
        expected_nvs = int(ctx["NvS"])
        assert src.dtype == torch.int32
        assert src.is_contiguous() and tuple(src.shape) == (expected_nvs,)
        assert src.device == topk_experts_flat.device
    torch.npu.synchronize()
    _trace(ctx, f"{impl_name}: rank core done")
    _copy_rank_outputs(ctx, outputs, cu_seqlens, plan)
    ctx["_planning_impl"] = impl_name
    ctx["_planning_launch_count"] = launch_count
    return src
