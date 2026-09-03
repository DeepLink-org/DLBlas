"""Self-contained multi-program Triton Ascend Planning implementation.

This file is the canonical reference implementation.  It keeps the reference
allocation/layout/route semantics and uses a destination-local direct-dst
deduplication kernel with a configurable number of programs.  The full kernel
definitions and dst canonicalization are intentionally local, with no import
from an archived reference or shared src-helper module.
The public return interface is ``(cu_seqlens, MoonEPCommPlan, src_or_none)``.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
import triton
import triton.language as tl

from .planning_ascend_common import (
    DEDUP_MODE_CURRENT,
    DEDUP_MODE_SRC,
    MoonEPCommPlan,
    PlanningGlobalOutputs,
    allocate_planning_outputs,
    normalize_dedup_mode,
    run_all_rank_planning,
)

IMPLEMENTATION_NAME = "triton_reference_semantic"


@triton.jit
def _reference_alloc_kernel(
    all_tpe,
    tpe_prefix,
    expert_count,
    balance,
    quotas,
    remaining,
    alloc,
    E,
    R,
    EPN,
    CAP,
):
    """Follow the reference allocation and lowest-index tie-break order."""
    for expert in tl.range(0, E):
        running = 0
        for source_rank in tl.range(0, R):
            count = tl.load(all_tpe + source_rank * E + expert, mask=True, other=0).to(
                tl.int32
            )
            running += count
            tl.store(tpe_prefix + source_rank * E + expert, running)
        tl.store(expert_count + expert, running)
        tl.store(remaining + expert, running)
        owner = expert // EPN
        for destination in tl.range(0, R):
            tl.store(
                alloc + destination * E + expert,
                tl.where(destination == owner, running, 0),
            )

    for index in tl.range(0, R * R):
        tl.store(quotas + index, 0)

    for home in tl.range(0, R):
        group_total = 0
        for local_expert in tl.range(0, EPN):
            group_total += tl.load(expert_count + home * EPN + local_expert)
        tl.store(balance + home, group_total - CAP)

    # At least one receiver is filled each active round, so R rounds suffice.
    for _ in tl.range(0, R):
        max_balance = -2147483648
        min_balance = 2147483647
        home = 0
        receiver = 0
        for rank in tl.range(0, R):
            value = tl.load(balance + rank)
            take_max = value > max_balance
            take_min = value < min_balance
            max_balance = tl.where(take_max, value, max_balance)
            min_balance = tl.where(take_min, value, min_balance)
            home = tl.where(take_max, rank, home)
            receiver = tl.where(take_min, rank, receiver)
        active = max_balance > 0
        move = tl.where(active, -min_balance, 0)
        tl.store(quotas + home * R + receiver, move, mask=active)
        tl.store(balance + home, max_balance - move, mask=active)
        tl.store(balance + receiver, 0, mask=active)

    for home in tl.range(0, R):
        base = home * EPN
        for _ in tl.range(0, EPN + R):
            max_quota = -1
            receiver = 0
            for destination in tl.range(0, R):
                value = tl.load(quotas + home * R + destination)
                better = value > max_quota
                max_quota = tl.where(better, value, max_quota)
                receiver = tl.where(better, destination, receiver)
            max_remaining = -1
            local_expert = 0
            for candidate in tl.range(0, EPN):
                value = tl.load(remaining + base + candidate)
                better = value > max_remaining
                max_remaining = tl.where(better, value, max_remaining)
                local_expert = tl.where(better, candidate, local_expert)
            active = max_quota > 0
            take = tl.where(active, tl.minimum(max_quota, max_remaining), 0)
            expert = base + local_expert
            remote_old = tl.load(alloc + receiver * E + expert, mask=active, other=0)
            home_old = tl.load(alloc + home * E + expert, mask=active, other=0)
            tl.store(
                alloc + receiver * E + expert,
                remote_old + take,
                mask=active,
            )
            tl.store(
                alloc + home * E + expert,
                home_old - take,
                mask=active,
            )
            tl.store(remaining + expert, max_remaining - take, mask=active)
            tl.store(
                quotas + home * R + receiver,
                max_quota - take,
                mask=active,
            )


@triton.jit
def _reference_layout_kernel(
    alloc,
    alloc_cumsum,
    expert_offsets,
    selected,
    stats,
    experts_to_copy,
    cu_seqlens,
    zero_fill_ranges,
    E,
    R,
    EPN,
    B,
    TOKEN_PADDING,
):
    """Materialize every rank's reference VM-group layout."""
    for index in tl.range(0, R * E):
        tl.store(expert_offsets + index, 0)
    for index in tl.range(0, R * 2):
        tl.store(stats + index, 0)
    for index in tl.range(0, R * B):
        tl.store(experts_to_copy + index, -1)
    for index in tl.range(0, R * (E + B)):
        tl.store(cu_seqlens + index, 0)
        tl.store(zero_fill_ranges + index * 2, 0)
        tl.store(zero_fill_ranges + index * 2 + 1, 0)

    for expert in tl.range(0, E):
        running = 0
        for destination in tl.range(0, R):
            running += tl.load(alloc + destination * E + expert)
            tl.store(alloc_cumsum + expert * R + destination, running)

    for destination in tl.range(0, R):
        local_start = destination * EPN
        local_end = local_start + EPN
        for expert in tl.range(0, E):
            tl.store(selected + expert, 0)
        remote_count = 0
        for expert in tl.range(0, E):
            count = tl.load(alloc + destination * E + expert)
            is_remote = (expert < local_start) | (expert >= local_end)
            remote_count += tl.where((count > 0) & is_remote, 1, 0)
        tl.store(stats + destination * 2, remote_count)

        for slot in tl.range(0, B):
            best_count = 0
            best_expert = -1
            for expert in tl.range(0, E):
                count = tl.load(alloc + destination * E + expert)
                used = tl.load(selected + expert)
                is_remote = (expert < local_start) | (expert >= local_end)
                candidate = (count > 0) & is_remote & (used == 0)
                better = candidate & (
                    (count > best_count)
                    | ((count == best_count) & (expert > best_expert))
                )
                best_count = tl.where(better, count, best_count)
                best_expert = tl.where(better, expert, best_expert)
            valid = best_expert >= 0
            tl.store(
                experts_to_copy + destination * B + slot,
                best_expert,
            )
            tl.store(selected + best_expert, 1, mask=valid)
            owner = best_expert // EPN
            old = tl.load(stats + owner * 2 + 1, mask=valid, other=0)
            tl.store(stats + owner * 2 + 1, old + 1, mask=valid)

        start = 0
        for group_idx in tl.range(0, E + B):
            is_base = group_idx < E
            prefetched = tl.load(selected + group_idx, mask=is_base, other=0)
            slot_expert = tl.load(
                experts_to_copy + destination * B + group_idx - E,
                mask=~is_base,
                other=-1,
            )
            base_active = is_base & (prefetched == 0)
            slot_active = (~is_base) & (slot_expert >= 0)
            expert = tl.where(
                base_active,
                group_idx,
                tl.where(slot_active, slot_expert, -1),
            )
            active = expert >= 0
            count = tl.load(
                alloc + destination * E + expert,
                mask=active,
                other=0,
            )
            padded = tl.where(
                count > 0,
                ((count + TOKEN_PADDING - 1) // TOKEN_PADDING) * TOKEN_PADDING,
                0,
            )
            end = start + padded
            output_index = destination * (E + B) + group_idx
            tl.store(cu_seqlens + output_index, end)
            tl.store(
                expert_offsets + destination * E + expert,
                start,
                mask=count > 0,
            )
            pad_count = padded - count
            tl.store(
                zero_fill_ranges + output_index * 2,
                tl.where(pad_count > 0, start + count, 0),
            )
            tl.store(
                zero_fill_ranges + output_index * 2 + 1,
                pad_count,
            )
            start = end


@triton.jit
def _reference_dst_kernel(
    topk,
    tpe_prefix,
    alloc_cumsum,
    expert_offsets,
    counters,
    dst,
    src,
    E,
    R,
    N,
    NVS,
    SOURCE_RANK,
    DESTINATION_RANK,
    WRITE_SRC: tl.constexpr,
):
    """Compute one source rank's raw dst in stable local occurrence order."""
    for expert in tl.range(0, E):
        tl.store(counters + expert, 0)
    for index in tl.range(0, N):
        expert = tl.load(topk + index).to(tl.int32)
        local_index = tl.load(counters + expert)
        tl.store(counters + expert, local_index + 1)
        previous_tpe = tl.load(
            tpe_prefix + (SOURCE_RANK - 1) * E + expert,
            mask=SOURCE_RANK > 0,
            other=0,
        )
        global_index = previous_tpe + local_index

        destination = 0
        previous_alloc = 0
        running_previous = 0
        chosen = 0
        for candidate in tl.range(0, R):
            cumulative = tl.load(alloc_cumsum + expert * R + candidate)
            take = (chosen == 0) & (cumulative > global_index)
            destination = tl.where(take, candidate, destination)
            previous_alloc = tl.where(take, running_previous, previous_alloc)
            chosen = tl.where(take, 1, chosen)
            running_previous = cumulative
        base = tl.load(expert_offsets + destination * E + expert)
        destination_loff = base + global_index - previous_alloc
        tl.store(dst + index, destination * NVS + destination_loff)
        if WRITE_SRC:
            tl.store(
                src + destination_loff,
                SOURCE_RANK * NVS + index,
                mask=destination == DESTINATION_RANK,
            )


@triton.jit
def _canonicalize_dst_kernel(
    dst,
    R: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NVS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Keep the first kidx per destination and negative-encode later ones."""
    source_rank = tl.program_id(0)
    token = tl.program_id(1)
    lane = tl.arange(0, BLOCK_K)
    valid = lane < K
    token_base = source_rank * N + token * K
    raw = tl.load(dst + token_base + lane, mask=valid, other=0)
    destination = raw // NVS
    duplicate = lane < 0
    for prior_kidx in tl.static_range(0, K):
        prior = tl.sum(
            tl.load(
                dst + token_base + prior_kidx + lane,
                mask=lane == 0,
                other=0,
            ),
            axis=0,
        )
        duplicate |= (lane > prior_kidx) & (destination == prior // NVS)
    tl.store(
        dst + token_base + lane,
        tl.where(duplicate, -raw - 1, raw),
        mask=valid,
    )


def canonicalize_dst(
    dst: torch.Tensor,
    *,
    R: int,
    S: int,
    K: int,
    N: int,
    NvS: int,
) -> int:
    """Launch local dst canonicalization without external helper modules."""
    block_k = max(8, triton.next_power_of_2(K))
    _canonicalize_dst_kernel[(R, S)](
        dst,
        R=R,
        S=S,
        K=K,
        N=N,
        NVS=NvS,
        BLOCK_K=block_k,
    )
    return 1


@triton.jit
def _local_direct_dst_dedup_multi_kernel(
    dst,
    dup_groups,
    dup_loffs,
    dup_counts,
    R: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NVS: tl.constexpr,
    TARGET_RANK: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Build one rank's compact dedup plan with NUM_SMS programs.

    Every key is handled by exactly one program.  Atomic reservation makes
    group and duplicate ranges disjoint; consumers compare groups
    semantically and do not require a stable allocation order.
    """
    pid = tl.program_id(0)

    # Striding over keys lets NUM_SMS directly control the number of active
    # programs without requiring a second prefix-scan kernel.
    for key in tl.range(pid, NUM_KEYS, NUM_SMS):
        source_rank = key // S
        token = key - source_rank * S
        token_base = source_rank * N + token * K
        entry_count = 0
        primary_loff = -1

        # Count entries targeting this destination and elect lowest kidx.
        for kidx in tl.static_range(0, K):
            raw = tl.load(dst + token_base + kidx)
            destination = raw // NVS
            local_offset = raw % NVS
            is_target = destination == TARGET_RANK
            primary_loff = tl.where(
                is_target & (entry_count == 0), local_offset, primary_loff
            )
            entry_count += tl.where(is_target, 1, 0)

        duplicate_count = tl.maximum(entry_count - 1, 0)
        valid_group = duplicate_count > 0

        # ``dup_counts`` is zero-initialized by allocate_global_outputs.  Each
        # valid key atomically reserves one group row and its duplicate prefix.
        if valid_group:
            group_index = tl.atomic_add(dup_counts, 1)
            duplicate_start = tl.atomic_add(dup_counts + 1, duplicate_count)
            tl.store(dup_groups + group_index * 3, primary_loff)
            tl.store(dup_groups + group_index * 3 + 1, duplicate_start)
            tl.store(dup_groups + group_index * 3 + 2, duplicate_count)

            # Emit duplicate local offsets in increasing kidx order.
            stored = 0
            seen_primary = 0
            for kidx in tl.static_range(0, K):
                raw = tl.load(dst + token_base + kidx)
                destination = raw // NVS
                local_offset = raw % NVS
                is_target = destination == TARGET_RANK
                is_primary = is_target & (seen_primary == 0)
                is_duplicate = is_target & (seen_primary != 0)
                tl.store(
                    dup_loffs + duplicate_start + stored,
                    local_offset,
                    mask=is_duplicate,
                )
                stored += tl.where(is_duplicate, 1, 0)
                seen_primary = tl.where(is_primary, 1, seen_primary)


def _rank_reference(
    ctx: dict,
    all_topk: torch.Tensor,
    all_tpe: torch.Tensor,
    outputs: PlanningGlobalOutputs,
    dedup_mode: str,
    src: torch.Tensor | None,
) -> int:
    """Run reference stages and NUM_SMS-controlled local direct-dst dedup."""
    R = int(ctx["R"])
    E = int(ctx["E"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    B = int(ctx["B"])
    N = int(ctx["N"])
    NvS = int(ctx["NvS"])
    capacity = int(ctx["NvS_capacity"])
    token_padding = int(ctx["token_padding"])
    epn = E // R
    device = all_tpe.device

    # These scratch tensors and the first three kernels mirror reference.
    tpe_prefix = torch.empty((R, E), dtype=torch.int32, device=device)
    expert_count = torch.empty(E, dtype=torch.int32, device=device)
    balance = torch.empty(R, dtype=torch.int32, device=device)
    quotas = torch.empty((R, R), dtype=torch.int32, device=device)
    remaining = torch.empty(E, dtype=torch.int32, device=device)
    alloc = torch.empty((R, E), dtype=torch.int32, device=device)
    alloc_cumsum = torch.empty((E, R), dtype=torch.int32, device=device)
    expert_offsets = torch.empty((R, E), dtype=torch.int32, device=device)
    selected = torch.empty(E, dtype=torch.int32, device=device)
    counters = torch.empty(E, dtype=torch.int32, device=device)

    _reference_alloc_kernel[(1,)](
        all_tpe,
        tpe_prefix,
        expert_count,
        balance,
        quotas,
        remaining,
        alloc,
        E,
        R,
        epn,
        capacity,
    )
    _reference_layout_kernel[(1,)](
        alloc,
        alloc_cumsum,
        expert_offsets,
        selected,
        outputs.remote_stats,
        outputs.experts_to_copy,
        outputs.cu_seqlens,
        outputs.zero_fill_ranges,
        E,
        R,
        epn,
        B,
        token_padding,
    )
    for source_rank in range(R):
        _reference_dst_kernel[(1,)](
            all_topk[source_rank],
            tpe_prefix,
            alloc_cumsum,
            expert_offsets,
            counters,
            outputs.dst[source_rank],
            src if src is not None else outputs.dst[source_rank],
            E,
            R,
            N,
            NvS,
            source_rank,
            int(ctx["rank"]),
            WRITE_SRC=dedup_mode == DEDUP_MODE_SRC,
        )

    rank = int(ctx["rank"])
    if dedup_mode in (DEDUP_MODE_CURRENT, DEDUP_MODE_SRC):
        # The source tensor is still produced in src mode, but dedup is built
        # directly from raw global dst to avoid reverse-index materialization.
        num_sms = max(1, int(ctx["num_sms"]))
        _local_direct_dst_dedup_multi_kernel[(num_sms,)](
            outputs.dst,
            outputs.dup_groups[rank],
            outputs.dup_loffs[rank],
            outputs.dup_counts[rank],
            R=R,
            S=S,
            K=K,
            N=N,
            NVS=NvS,
            TARGET_RANK=rank,
            NUM_KEYS=R * S,
            NUM_SMS=num_sms,
        )
        post_launches = canonicalize_dst(
            outputs.dst,
            R=R,
            S=S,
            K=K,
            N=N,
            NvS=NvS,
        )
        post_launches += 1
    else:
        post_launches = canonicalize_dst(
            outputs.dst,
            R=R,
            S=S,
            K=K,
            N=N,
            NvS=NvS,
        )

    return 2 + R + post_launches


def launch_planning(
    ctx: dict,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    dedup_mode: str = DEDUP_MODE_CURRENT,
) -> tuple[torch.Tensor, MoonEPCommPlan, torch.Tensor | None]:
    """Allocate and return rank-local outputs for the update1 variant."""
    dedup_mode = normalize_dedup_mode(dedup_mode)
    expected_topk_shape = (int(ctx["S"]), int(ctx["K"]))
    if tuple(topk_experts.shape) != expected_topk_shape:
        raise ValueError(
            f"topk_experts must have shape {expected_topk_shape}, "
            f"got {tuple(topk_experts.shape)}"
        )

    plan, cu_seqlens = allocate_planning_outputs(ctx)
    src = run_all_rank_planning(
        IMPLEMENTATION_NAME,
        _rank_reference,
        ctx,
        topk_experts.view(-1),
        tokens_per_expert,
        cu_seqlens,
        plan,
        dedup_mode=dedup_mode,
    )
    return cu_seqlens, plan, src
