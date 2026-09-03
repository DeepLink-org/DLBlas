"""Faithful Torch/NPU port of MoonEP's Planning correctness reference."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from .planning_ascend_common import (
    DEDUP_MODE_CURRENT,
    DEDUP_MODE_SRC,
    DEDUP_MODE_ZERO,
    MoonEPCommPlan,
    PlanningGlobalOutputs,
    allocate_planning_outputs,
    normalize_dedup_mode,
    run_all_rank_planning,
)


IMPLEMENTATION_NAME = "torch_reference_port"


def _build_allocation(ctx: dict, tpe: torch.Tensor):
    """Port the official token-prefix and deterministic balancing stages."""
    R = int(ctx["R"])
    E = int(ctx["E"])
    epn = E // R
    capacity = int(ctx["NvS_capacity"])
    device = tpe.device

    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[R - 1]
    group_tokens = torch.zeros(R, dtype=torch.int32, device=device)
    for home in range(R):
        group_tokens[home] = expert_count[home * epn : (home + 1) * epn].sum()
    balance = group_tokens - capacity

    alloc = torch.zeros((E, R), dtype=torch.int32, device=device)
    for expert in range(E):
        alloc[expert, expert // epn] = expert_count[expert]

    # z[home, destination] follows the exact reference selection policy.
    z = torch.zeros((R, R), dtype=torch.int32, device=device)
    while True:
        home = int(balance.argmax().item())
        destination = int(balance.argmin().item())
        if int(balance[home].item()) <= 0:
            break
        move = -int(balance[destination].item())
        if move <= 0:
            raise AssertionError("positive load has no receiver capacity")
        z[home, destination] = move
        balance[home] -= move
        balance[destination] = 0

    for home in range(R):
        expert_start = home * epn
        remaining = expert_count[expert_start : expert_start + epn].clone()
        quotas = z[home].clone()
        while True:
            destination = int(quotas.argmax().item())
            quota = int(quotas[destination].item())
            if quota <= 0:
                break
            local_expert = int(remaining.argmax().item())
            expert = expert_start + local_expert
            rem = int(remaining[local_expert].item())
            take = min(rem, quota)
            if take <= 0:
                raise AssertionError("remote quota cannot be filled")
            alloc[expert, destination] += take
            alloc[expert, home] -= take
            remaining[local_expert] -= take
            quotas[destination] -= take

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError("per-expert token conservation failed")
    if bool((alloc.sum(dim=0) > capacity).any().item()):
        raise AssertionError("rank capacity exceeded")
    return tpe_cumsum, alloc


def _build_layout(
    ctx: dict,
    alloc: torch.Tensor,
    outputs: PlanningGlobalOutputs,
):
    """Port the official VM-group layout and prefetch selection stages."""
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    epn = E // R
    token_padding = int(ctx["token_padding"])
    NvS = int(ctx["NvS"])
    device = alloc.device
    expert_off = torch.zeros((R, E), dtype=torch.int32, device=device)

    for destination in range(R):
        local_start = destination * epn
        local_end = local_start + epn
        remote_experts = [
            expert
            for expert in range(E)
            if int(alloc[expert, destination].item()) > 0
            and not (local_start <= expert < local_end)
        ]
        remote_experts.sort(
            key=lambda expert: (
                int(alloc[expert, destination].item()),
                expert,
            ),
            reverse=True,
        )
        outputs.remote_stats[destination, 0] = len(remote_experts)
        for slot, expert in enumerate(remote_experts[:B]):
            outputs.experts_to_copy[destination, slot] = expert
            outputs.remote_stats[expert // epn, 1] += 1

        experts_to_prefetch = set(remote_experts[:B])
        start = 0
        for group_idx in range(E + B):
            count = 0
            expert_id = -1
            if group_idx < E:
                if group_idx not in experts_to_prefetch:
                    expert_id = group_idx
                    count = int(alloc[expert_id, destination].item())
            else:
                slot = group_idx - E
                expert_id = int(outputs.experts_to_copy[destination, slot].item())
                if expert_id >= 0:
                    count = int(alloc[expert_id, destination].item())

            padded = (
                ((count + token_padding - 1) // token_padding) * token_padding
                if count > 0
                else 0
            )
            aligned_end = start + padded
            outputs.cu_seqlens[destination, group_idx] = aligned_end
            if count > 0:
                expert_off[destination, expert_id] = start
                pad_count = padded - count
                if pad_count > 0:
                    outputs.zero_fill_ranges[destination, group_idx, 0] = start + count
                    outputs.zero_fill_ranges[destination, group_idx, 1] = pad_count
            start = aligned_end
        if start > NvS:
            raise AssertionError("padded layout exceeds NvS")
    return expert_off


def _build_raw_dst(
    ctx: dict,
    topk: torch.Tensor,
    tpe_cumsum: torch.Tensor,
    alloc: torch.Tensor,
    expert_off: torch.Tensor,
    outputs: PlanningGlobalOutputs,
    src: torch.Tensor | None,
) -> None:
    """Write raw dst and, in src mode, this destination rank's src_info."""
    R = int(ctx["R"])
    E = int(ctx["E"])
    N = int(ctx["N"])
    NvS = int(ctx["NvS"])
    device = topk.device
    alloc_cumsum = alloc.cumsum(dim=1)

    for source_rank in range(R):
        counters = torch.zeros(E, dtype=torch.int32, device=device)
        flat_topk = topk[source_rank].reshape(-1)
        for index in range(N):
            expert = int(flat_topk[index].item())
            local_index = int(counters[expert].item())
            counters[expert] += 1
            previous_tpe = (
                0
                if source_rank == 0
                else int(tpe_cumsum[source_rank - 1, expert].item())
            )
            global_index = previous_tpe + local_index

            destination = 0
            while (
                destination < R
                and int(alloc_cumsum[expert, destination].item()) <= global_index
            ):
                destination += 1
            if destination >= R:
                raise AssertionError("no destination rank found")
            previous_alloc = (
                0
                if destination == 0
                else int(alloc_cumsum[expert, destination - 1].item())
            )
            destination_loff = (
                int(expert_off[destination, expert].item())
                + global_index
                - previous_alloc
            )
            outputs.dst[source_rank, index] = destination * NvS + destination_loff
            if src is not None and destination == int(ctx["rank"]):
                if int(src[destination_loff].item()) != -1:
                    raise AssertionError("src destination collision")
                src[destination_loff] = source_rank * NvS + index


def _build_dedup(ctx: dict, outputs: PlanningGlobalOutputs) -> None:
    """Port canonical dst encoding and deterministic destination-owned plans."""
    R = int(ctx["R"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    device = outputs.dst.device

    for source_rank in range(R):
        for token in range(S):
            base = token * K
            dst_values = outputs.dst[source_rank, base : base + K]
            destinations = torch.div(dst_values, NvS, rounding_mode="floor")
            local_offsets = dst_values % NvS
            groups: dict[int, list[int]] = {}
            indices: dict[int, list[int]] = {}
            for k_index in range(K):
                destination = int(destinations[k_index].item())
                groups.setdefault(destination, []).append(
                    int(local_offsets[k_index].item())
                )
                indices.setdefault(destination, []).append(k_index)

            for destination, group_offsets in groups.items():
                duplicate_count = len(group_offsets) - 1
                if duplicate_count > 0:
                    group_index = int(outputs.dup_counts[destination, 0].item())
                    duplicate_start = int(outputs.dup_counts[destination, 1].item())
                    outputs.dup_groups[destination, group_index, 0] = group_offsets[0]
                    outputs.dup_groups[destination, group_index, 1] = duplicate_start
                    outputs.dup_groups[destination, group_index, 2] = duplicate_count
                    outputs.dup_loffs[
                        destination,
                        duplicate_start : duplicate_start + duplicate_count,
                    ] = torch.tensor(
                        group_offsets[1:], dtype=torch.int32, device=device
                    )
                    outputs.dup_counts[destination, 0] += 1
                    outputs.dup_counts[destination, 1] += duplicate_count

                for duplicate_index in range(1, len(group_offsets)):
                    flat_index = base + indices[destination][duplicate_index]
                    outputs.dst[source_rank, flat_index] = (
                        -outputs.dst[source_rank, flat_index] - 1
                    )


def _canonicalize_dst(ctx: dict, outputs: PlanningGlobalOutputs) -> None:
    """Canonicalize dst without materializing destination dedup structures."""
    R = int(ctx["R"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    for source_rank in range(R):
        for token in range(S):
            base = token * K
            seen = set()
            for kidx in range(K):
                index = base + kidx
                raw_dst = int(outputs.dst[source_rank, index].item())
                destination = raw_dst // NvS
                if destination in seen:
                    outputs.dst[source_rank, index] = -raw_dst - 1
                else:
                    seen.add(destination)


def _build_dedup_from_src(
    ctx: dict,
    outputs: PlanningGlobalOutputs,
    src: torch.Tensor,
) -> None:
    """Build this rank's deterministic dedup plan from MoonEP src_info."""
    rank = int(ctx["rank"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    entries_by_key: dict[int, dict[int, int]] = {}
    for local_offset in range(NvS):
        src_value = int(src[local_offset].item())
        if src_value < 0:
            continue
        source_rank = src_value // NvS
        offv = src_value - source_rank * NvS
        token = offv // K
        kidx = offv - token * K
        key = source_rank * S + token
        entries = entries_by_key.setdefault(key, {})
        if kidx in entries:
            raise AssertionError("duplicate src kidx")
        entries[kidx] = local_offset

    groups = []
    for entries in entries_by_key.values():
        ordered = sorted(entries.items())
        if len(ordered) > 1:
            groups.append((ordered[0][1], [item[1] for item in ordered[1:]]))
    groups.sort(key=lambda item: item[0])

    duplicate_start = 0
    for group_index, (primary_loff, duplicate_loffs) in enumerate(groups):
        duplicate_count = len(duplicate_loffs)
        outputs.dup_groups[rank, group_index, 0] = primary_loff
        outputs.dup_groups[rank, group_index, 1] = duplicate_start
        outputs.dup_groups[rank, group_index, 2] = duplicate_count
        outputs.dup_loffs[rank, duplicate_start : duplicate_start + duplicate_count] = (
            torch.tensor(duplicate_loffs, dtype=torch.int32, device=src.device)
        )
        duplicate_start += duplicate_count
    outputs.dup_counts[rank, 0] = len(groups)
    outputs.dup_counts[rank, 1] = duplicate_start


def _rank_torch_port(
    ctx: dict,
    all_topk: torch.Tensor,
    all_tpe: torch.Tensor,
    outputs: PlanningGlobalOutputs,
    dedup_mode: str,
    src: torch.Tensor | None,
) -> int:
    """Execute the complete faithful reference port on the current rank."""
    tpe_cumsum, alloc = _build_allocation(ctx, all_tpe)
    expert_off = _build_layout(ctx, alloc, outputs)
    _build_raw_dst(ctx, all_topk, tpe_cumsum, alloc, expert_off, outputs, src)
    if dedup_mode == DEDUP_MODE_CURRENT:
        _build_dedup(ctx, outputs)
    elif dedup_mode == DEDUP_MODE_ZERO:
        _canonicalize_dst(ctx, outputs)
    else:
        assert dedup_mode == DEDUP_MODE_SRC and src is not None
        _canonicalize_dst(ctx, outputs)
        _build_dedup_from_src(ctx, outputs, src)
    return 1


def launch_planning(
    ctx: dict,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    dedup_mode: str = DEDUP_MODE_CURRENT,
) -> tuple[torch.Tensor, MoonEPCommPlan, torch.Tensor | None]:
    """Run Planning from rank-local inputs and return local output objects."""
    dedup_mode = normalize_dedup_mode(dedup_mode)
    expected_topk_shape = (int(ctx["S"]), int(ctx["K"]))
    if tuple(topk_experts.shape) != expected_topk_shape:
        raise ValueError(
            f"topk_experts must have shape {expected_topk_shape}, "
            f"got {tuple(topk_experts.shape)}"
        )

    plan, cu_seqlens = allocate_planning_outputs(ctx)
    topk_experts_flat = topk_experts.reshape(-1)
    src = run_all_rank_planning(
        IMPLEMENTATION_NAME,
        _rank_torch_port,
        ctx,
        topk_experts_flat,
        tokens_per_expert,
        cu_seqlens,
        plan,
        dedup_mode=dedup_mode,
    )
    return cu_seqlens, plan, src
