"""Standalone cases, multi-rank fixtures, oracle, comparisons, and invariants."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the NPU backend

from .planning_ascend_common import (
    DEDUP_MODE_CURRENT,
    DEDUP_MODE_SRC,
    DEDUP_MODE_ZERO,
    DEDUP_MODES,
    normalize_dedup_mode,
)

DEFAULT_TOKEN_PADDING = 128


@dataclass(frozen=True)
class KernelCase:
    """One MoonEP-style Planning test configuration."""

    name: str
    S: int
    K: int
    epn: int
    H: int
    num_sms: int
    B: int | None = None
    token_padding: int = DEFAULT_TOKEN_PADDING
    routing: str = "balanced"
    bias_ratio: float = 0.0
    seed: int = 42
    min_R: int = 1
    max_R: int | None = None

    def E(self, R: int) -> int:
        """Return the global expert count for a world size."""
        return R * self.epn


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
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

    # Dedup structures written by the dispatch builder and consumed by the
    # dispatch epilogue / combine prologue. dup_counts = [n_groups, n_dup_loffs];
    # only the compact prefix of dup_groups/dup_loffs is valid, and the ordering
    # is decided by the builder's atomicAdd — not guaranteed to be stable.
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the official plan fields, shapes, dtypes, and contiguity."""
        N = int(self.N)
        R = int(self.R)
        E = int(self.E)
        B = int(self.B)
        NvS = int(self.NvS)
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()
        assert self.dst.numel() == N
        assert (
            self.experts_to_copy.dtype == torch.int32
            and self.experts_to_copy.is_contiguous()
        )
        assert tuple(self.experts_to_copy.shape) == (R, B)
        assert (
            self.remote_stats.dtype == torch.int32 and self.remote_stats.is_contiguous()
        )
        assert tuple(self.remote_stats.shape) == (2,)
        assert (
            self.zero_fill_ranges.dtype == torch.int32
            and self.zero_fill_ranges.is_contiguous()
        )
        assert tuple(self.zero_fill_ranges.shape) == (E + B, 2)
        assert self.dup_groups.dtype == torch.int32 and self.dup_groups.is_contiguous()
        assert tuple(self.dup_groups.shape) == (NvS, 3)
        assert self.dup_loffs.dtype == torch.int32 and self.dup_loffs.is_contiguous()
        assert tuple(self.dup_loffs.shape) == (NvS,)
        assert self.dup_counts.dtype == torch.int32 and self.dup_counts.is_contiguous()
        assert tuple(self.dup_counts.shape) == (2,)

    def clone(self) -> "MoonEPCommPlan":
        """Return an independent copy of every plan tensor and scalar field."""
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N,
            R=self.R,
            E=self.E,
            B=self.B,
            NvS=self.NvS,
            K=self.K,
        )


@dataclass(frozen=True)
class ReferenceDedupPlan:
    """Reference dup structures for the dispatch builder.

    The production builder allocates the compact prefixes with per-warp
    atomicAdds, so its ``dup_groups`` / ``dup_loffs`` ordering is not stable
    run-to-run; this reference uses a deterministic order and tests must
    compare group *sets*, never element order.
    """

    dup_groups: torch.Tensor  # [NvS, 3] (primary_loff, dup_start, dup_n)
    dup_loffs: torch.Tensor  # [NvS]
    dup_counts: torch.Tensor  # [2] (n_groups, n_dup_loffs)


PLANNING_CASES = [
    # Default benchmark setting for the Ascend 910B reference path.
    KernelCase("balanced_epn16", 256, 8, 16, 128, 24),
    KernelCase("tiny_s1_k1", 1, 1, 1, 16, 1, token_padding=8),
    KernelCase(
        "tiny_biased_with_prefetch",
        3,
        5,
        3,
        16,
        1,
        B=1,
        token_padding=32,
        routing="biased",
        bias_ratio=1.0,
        seed=9002,
        min_R=2,
    ),
    KernelCase(
        "no_padding",
        33,
        2,
        2,
        16,
        1,
        B=1,
        token_padding=1,
        routing="biased",
        bias_ratio=1.0,
        seed=10001,
    ),
    KernelCase(
        "non_power_mild_bias",
        17,
        3,
        3,
        16,
        1,
        B=2,
        token_padding=16,
        routing="biased",
        bias_ratio=0.5,
        seed=10501,
    ),
    KernelCase(
        "small_balanced_with_prefetch",
        17,
        3,
        3,
        16,
        1,
        B=2,
        token_padding=16,
        seed=11001,
    ),
    KernelCase(
        "near_degenerate_bias",
        64,
        2,
        5,
        32,
        32,
        B=3,
        routing="biased",
        bias_ratio=5.0,
        seed=12001,
    ),
    KernelCase(
        "typical_bias",
        32,
        4,
        4,
        32,
        32,
        B=2,
        token_padding=64,
        routing="biased",
        bias_ratio=1.0,
        seed=13001,
    ),
    KernelCase(
        "heavy_bias",
        96,
        3,
        7,
        48,
        1,
        B=4,
        routing="biased",
        bias_ratio=2.0,
        seed=14001,
    ),
    KernelCase(
        "step1_cross_cta_group",
        1024,
        1,
        256,
        16,
        8,
        B=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_multi_chunk_full_tile",
        2048,
        1,
        320,
        16,
        1,
        B=1,
        token_padding=1,
    ),
    KernelCase(
        "step1_segment_tail_full_tile",
        1536,
        1,
        320,
        16,
        2,
        B=1,
        token_padding=1,
        min_R=4,
    ),
    KernelCase(
        "experts_gt_block_size",
        256,
        1,
        1025,
        16,
        8,
        B=1,
        token_padding=1,
        max_R=2,
    ),
    KernelCase(
        "all_local",
        32,
        4,
        8,
        32,
        8,
        token_padding=16,
        routing="all_local",
    ),
    KernelCase(
        "all_remote_with_prefetch_slots",
        32,
        4,
        8,
        32,
        8,
        B=3,
        token_padding=16,
        routing="all_remote",
        min_R=2,
    ),
    KernelCase(
        "duplicate_topk",
        16,
        4,
        4,
        32,
        8,
        B=2,
        token_padding=8,
        routing="duplicate_topk",
        min_R=2,
    ),
    KernelCase(
        "single_expert_exact_padding",
        8,
        2,
        4,
        16,
        1,
        B=1,
        token_padding=8,
        routing="single_expert",
    ),
    KernelCase(
        "single_expert_padding_tail",
        9,
        2,
        4,
        16,
        1,
        B=1,
        token_padding=8,
        routing="single_expert",
    ),
]


STANDALONE_CASES = [
    PLANNING_CASES[0],
    PLANNING_CASES[1],
    PLANNING_CASES[7],
    PLANNING_CASES[14],
    PLANNING_CASES[15],
    PLANNING_CASES[17],
]


def case_params(cases):
    """Give shared cases stable pytest identifiers."""
    return [pytest.param(case, id=case.name) for case in cases]


def npu_index_for_local_rank(local_rank: int, device_ids: list[int]) -> int:
    """Resolve a torch-npu index under optional Ascend visible-device remapping."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if visible:
        visible_ids = [int(value.strip()) for value in visible.split(",")]
        if visible_ids != device_ids:
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES must match MOONEP_NPU_IDS: "
                f"visible={visible_ids}, requested={device_ids}"
            )
        # Ascend remaps the selected physical IDs to logical indices 0..R-1.
        return local_rank
    return device_ids[local_rank]


def skip_if_unsupported_world_size(case: KernelCase, R: int) -> None:
    """Apply only the official case world-size restrictions."""
    if R < case.min_R:
        pytest.skip(f"case {case.name} requires R >= {case.min_R}, got R={R}")
    if case.max_R is not None and R > case.max_R:
        pytest.skip(f"case {case.name} requires R <= {case.max_R}, got R={R}")


def init_case(case: KernelCase, rank: int, R: int) -> dict:
    """Create the minimal rank-local context accepted by every implementation."""
    skip_if_unsupported_world_size(case, R)
    E = case.E(R)
    B = case.epn if case.B is None else case.B
    N = case.S * case.K
    NvS_capacity = N
    NvS = NvS_capacity + (case.token_padding - 1) * 2 * case.epn
    return {
        "rank": int(rank),
        "R": int(R),
        "group": dist.group.WORLD,
        "S": int(case.S),
        "K": int(case.K),
        "E": int(E),
        "B": int(B),
        "N": int(N),
        "NvS_capacity": int(NvS_capacity),
        "NvS": int(NvS),
        "token_padding": int(case.token_padding),
        "num_sms": int(case.num_sms),
    }


def make_topk(case: KernelCase, rank: int, R: int):
    """Generate one physical rank's local routing and histogram on NPU."""
    skip_if_unsupported_world_size(case, R)
    E = case.E(R)
    if case.K > E and case.routing in {"balanced", "biased"}:
        pytest.skip(f"case {case.name} requires K <= E, got K={case.K}, E={E}")

    if case.routing in {"balanced", "biased"}:
        shared = torch.Generator(device="cpu").manual_seed(case.seed)
        local = torch.Generator(device="cpu").manual_seed(rank)
        if case.bias_ratio == 0.0:
            tokens = torch.arange(case.S)
            slots = torch.arange(case.K)
            target_rank = (tokens[:, None] + slots[None, :]) % R
            target_local = ((tokens[:, None] // R) + slots[None, :]) % case.epn
            permutation = torch.randperm(case.epn, generator=local)
            topk = target_rank * case.epn + permutation[target_local]
        else:
            logits = torch.exp(
                torch.normal(
                    0.0,
                    case.bias_ratio,
                    size=(E,),
                    generator=shared,
                )
            )
            topk = torch.multinomial(
                logits[None, :].expand(case.S, E),
                case.K,
                replacement=False,
                generator=local,
            )
    else:
        tokens = torch.arange(case.S)[:, None]
        slots = torch.arange(case.K)[None, :]
        if case.routing == "all_local":
            topk = rank * case.epn + ((tokens + slots) % case.epn)
        elif case.routing == "all_remote":
            remote_rank = (rank + 1) % R
            topk = remote_rank * case.epn + ((tokens + slots) % case.epn)
        elif case.routing == "single_expert":
            topk = torch.zeros((case.S, case.K), dtype=torch.long)
        elif case.routing == "duplicate_topk":
            expert = ((rank + 1) % R) * case.epn
            topk = torch.full((case.S, case.K), expert, dtype=torch.long)
        else:
            raise ValueError(f"unknown routing pattern: {case.routing}")

    topk = topk.to(torch.int32).contiguous()
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    device = torch.device("npu", torch.npu.current_device())
    return topk.to(device), tpe.to(device)


def launch_planning_torch_reference(
    ctx,
    topk_experts,
    tokens_per_expert,
    *,
    return_alloc=False,
    dedup_mode=DEDUP_MODE_CURRENT,
):
    """Compute the independent official-style Planning oracle for one rank."""
    dedup_mode = normalize_dedup_mode(dedup_mode)
    rank = ctx["rank"]
    R = ctx["R"]
    E = ctx["E"]
    B = int(ctx["B"])
    S = ctx["S"]
    K = ctx["K"]
    N = S * K
    epn = E // R
    CAP = ctx["NvS_capacity"]
    NvS = ctx.get("NvS", CAP)  # TODO: a bit odd; this lets scans omit NvS
    token_padding = ctx.get("token_padding", 1)
    group = ctx.get("group")

    flat_topk_experts = (
        topk_experts[rank].reshape(-1)
        if topk_experts.dim() == 3
        else topk_experts.reshape(-1)
    )
    if tokens_per_expert.dim() == 2:
        tpe = tokens_per_expert
    else:
        tpe = tokens_per_expert.reshape(1, E)
        if R > 1:
            gathered = [torch.zeros_like(tokens_per_expert) for _ in range(R)]
            dist.all_gather(gathered, tokens_per_expert, group=group)
            tpe = torch.stack(gathered)
    flat_topk_experts = flat_topk_experts.cpu()
    tpe = tpe.cpu()

    # tpe[src_rank, expert] is the token count of each expert on each source
    # rank. tpe_cumsum prefixes over source ranks; it later converts the
    # per-rank expert token index into a global one. The global sequence is
    # concatenated in source-rank order.
    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[R - 1]

    group_tokens = torch.zeros(R, dtype=expert_count.dtype)
    for h in range(R):
        group_tokens[h] = expert_count[h * epn : (h + 1) * epn].sum()

    # CAP is the real token capacity of each destination rank. Every source
    # rank has S*K routed entries, so sum(balance) = 0; positive means the
    # home group is overloaded, negative means the destination rank has room.
    balance = group_tokens - CAP

    # alloc[e, d] is how many global tokens of expert e land on dest rank d.
    # Note: kernel-side ctx['alloc'] uses the transposed [R, E] layout
    # (alloc[d*E+e]) so step4 reads coalesced by dest rank; here we compute
    # internally in [E, R] (clearest for migration/balancing) and transpose
    # on return_alloc to match the kernel buffer.
    alloc = torch.zeros(E, R, dtype=torch.int32)
    for e in range(E):
        alloc[e, e // epn] = expert_count[e]

    # z[h, d] is the token count home group h migrates to remote dest rank d.
    z = torch.zeros(R, R, dtype=torch.int32)

    # Choose remote receivers and their receive quotas.
    while True:
        # Pick the most overloaded home group and the roomiest dest rank; the
        # policy fills receivers fully, so chosen ones become balance[u] = 0.
        h = balance.argmax()
        u = balance.argmin()
        if balance[h] <= 0:
            break

        move = -balance[u]
        z[h, u] = move
        balance[h] -= move
        balance[u] = 0

    # From the receive quotas, compute exactly which tokens go to which
    # remote rank.
    for h in range(R):
        expert_start = h * epn
        remaining = expert_count[expert_start : expert_start + epn].clone()
        quotas = z[h].clone()

        while True:
            # Each round, re-pick the largest remote quota and the local
            # expert with the most remaining tokens.
            d = quotas.argmax()
            quota = quotas[d]
            if quota <= 0:
                break

            local_e = remaining.argmax()
            e = expert_start + local_e
            rem = remaining[local_e]

            take = torch.minimum(rem, quota)
            alloc[e, d] += take
            alloc[e, h] -= take
            remaining[local_e] -= take
            quotas[d] -= take

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError(
            "torch planning reference: per-expert token conservation failed"
        )
    if bool((alloc.sum(dim=0) > CAP).any().item()):
        raise AssertionError("torch planning reference: rank capacity exceeded")

    if return_alloc:
        # Transpose to the kernel's [R, E] layout (flat as alloc[d*E + e]).
        return alloc.t().contiguous()

    # Planning lookup tables: build the tables Part 2 and dispatch/prefetch
    # will use.
    alloc_cumsum = alloc.cumsum(dim=1)

    expert_off = torch.zeros(R, E, dtype=torch.int32)
    cu_seqlens = torch.zeros(R, E + B, dtype=torch.int32)
    experts_to_copy = torch.full((R, B), -1, dtype=torch.int32)
    remote_stats_all = torch.zeros(R, 2, dtype=torch.int32)
    # zero_fill_ranges[d, g] = (pad_start_loff, n_pad_rows): the segment-
    # padding rows the dispatch zero warp clears. If n_pad_rows == 0, the
    # kernel leaves pad_start_loff as 0 because dispatch never reads it.
    # cnt == 0 groups occupy no space and contribute 0 pad rows; the
    # [last_padded_end, NvS) tail is intentionally NOT covered (master
    # behavior — consumers read via cu_seqlens, the tail is undefined).
    zero_fill_by_rank = torch.zeros(R, E + B, 2, dtype=torch.int32)

    for d in range(R):
        local_start = d * epn
        local_end = local_start + epn

        remote_experts = [
            e
            for e in range(E)
            if alloc[e, d].item() > 0 and not (local_start <= e < local_end)
        ]

        # Pick the B remote expert segments with the most tokens for VM
        # prefetch;
        remote_experts.sort(key=lambda e: (alloc[e, d].item(), e), reverse=True)
        remote_stats_all[d, 0] = len(remote_experts)
        for b, e in enumerate(remote_experts[:B]):
            experts_to_copy[d, b] = e
            remote_stats_all[e // epn, 1] += 1

        experts_to_prefetch = set(remote_experts[:B])
        start = 0

        # The physical token order follows the VM group id:
        #   0..E-1 are the global expert groups;
        #   E..E+B-1 are the prefetch slots of the selected remote experts.
        for g in range(E + B):
            cnt = 0
            expert_id = -1

            if g < E:
                if g not in experts_to_prefetch:
                    cnt = alloc[g, d].item()
                    expert_id = g
            else:
                b = g - E
                if experts_to_copy[d, b].item() >= 0:
                    expert_id = experts_to_copy[d, b].item()
                    cnt = alloc[expert_id, d].item()

            end = start + cnt
            if cnt > 0:
                padded = ((cnt + token_padding - 1) // token_padding) * token_padding
            else:
                padded = 0
            aligned_end = start + padded

            cu_seqlens[d, g] = aligned_end

            if cnt > 0:
                expert_off[d, expert_id] = start
                n_pad = padded - cnt
                if n_pad > 0:
                    zero_fill_by_rank[d, g, 0] = end
                    zero_fill_by_rank[d, g, 1] = n_pad

            start = aligned_end

        if cu_seqlens[d].max().item() > NvS:
            raise AssertionError("torch planning reference: padded layout exceeds NvS")

    # Part 2: route assignment. MoonEP Phase D writes the source rank's dst
    # and the destination rank's src_info in the same route operation. The
    # standalone reference mirrors that ordering in src mode without making
    # an extra raw-dst copy or a full [R, NvS] src tensor.
    src_cpu = (
        torch.full((NvS,), -1, dtype=torch.int32)
        if dedup_mode == DEDUP_MODE_SRC
        else None
    )

    def route_source(source_topk, source_rank, local_src=None):
        """Build one source rank's raw dst and optionally this rank's src."""
        source_topk = source_topk.reshape(-1)
        local_cnt = torch.zeros(N, dtype=torch.int32)
        counter = torch.zeros(E, dtype=torch.int32)
        for index in range(N):
            expert = int(source_topk[index].item())
            local_cnt[index] = counter[expert]
            counter[expert] += 1

        source_dst = torch.zeros(N, dtype=torch.int32)
        for offv in range(N):
            expert = int(source_topk[offv].item())
            previous_source = (
                0
                if source_rank == 0
                else int(tpe_cumsum[source_rank - 1, expert].item())
            )
            global_rank = previous_source + int(local_cnt[offv].item())
            dest_rank = int(
                torch.searchsorted(alloc_cumsum[expert], global_rank, right=True).item()
            )
            if dest_rank >= R:
                raise AssertionError(
                    "torch planning reference: no destination rank found"
                )
            previous_alloc = (
                0 if dest_rank == 0 else int(alloc_cumsum[expert, dest_rank - 1].item())
            )
            dest_loff = (
                int(expert_off[dest_rank, expert].item()) + global_rank - previous_alloc
            )
            raw_dst = dest_rank * NvS + dest_loff
            source_dst[offv] = raw_dst
            if local_src is not None and dest_rank == rank:
                if int(local_src[dest_loff].item()) != -1:
                    raise AssertionError(
                        "torch planning reference: src destination collision"
                    )
                local_src[dest_loff] = source_rank * NvS + offv
        return source_dst

    if dedup_mode == DEDUP_MODE_SRC:
        if topk_experts.dim() == 3:
            all_topk_cpu = topk_experts.reshape(R, N).cpu()
        elif R == 1:
            all_topk_cpu = flat_topk_experts.reshape(1, N)
        else:
            gathered_topk = [torch.zeros_like(topk_experts) for _ in range(R)]
            dist.all_gather(gathered_topk, topk_experts, group=group)
            all_topk_cpu = torch.stack(gathered_topk).reshape(R, N).cpu()
        dst = None
        for source_rank in range(R):
            source_dst = route_source(all_topk_cpu[source_rank], source_rank, src_cpu)
            if source_rank == rank:
                dst = source_dst
        assert dst is not None
    else:
        dst = route_source(flat_topk_experts, rank)

    def canonicalize_local_dst(local_dst):
        """Apply MoonEP's first-kidx-per-destination negative encoding."""
        for token in range(S):
            base = token * K
            seen = set()
            for kidx in range(K):
                index = base + kidx
                raw_dst = int(local_dst[index].item())
                dest_rank = raw_dst // NvS
                if dest_rank in seen:
                    local_dst[index] = -raw_dst - 1
                else:
                    seen.add(dest_rank)

    # Part 3: mode-specific dedup materialization. current preserves the
    # existing all-rank deterministic reference. zero leaves all tensors zero.
    # src reconstructs this destination rank's compact plan from src_info.
    dup_groups_by_rank = torch.zeros((R, NvS, 3), dtype=torch.int32)
    dup_loffs_by_rank = torch.zeros((R, NvS), dtype=torch.int32)
    dup_counts_by_rank = torch.zeros((R, 2), dtype=torch.int32)

    if dedup_mode == DEDUP_MODE_CURRENT:
        dst_by_rank = dst.unsqueeze(dim=0).contiguous()
        if R > 1:
            dev = tokens_per_expert.device
            gathered = [torch.zeros_like(dst, device=dev) for _ in range(R)]
            dist.all_gather(gathered, dst.to(device=dev), group=group)
            dst_by_rank = torch.stack(gathered).cpu()

        for source_rank in range(R):
            for token in range(S):
                base = token * K
                dst_values = dst_by_rank[source_rank, base : base + K]
                destinations = torch.div(dst_values, NvS, rounding_mode="floor")
                local_offsets = dst_values % NvS
                groups = {}
                indices = {}
                for kidx in range(K):
                    dest_rank = int(destinations[kidx].item())
                    groups.setdefault(dest_rank, []).append(
                        int(local_offsets[kidx].item())
                    )
                    indices.setdefault(dest_rank, []).append(kidx)

                for dest_rank, group_loffs in groups.items():
                    duplicate_count = len(group_loffs) - 1
                    if duplicate_count > 0:
                        group_index = int(dup_counts_by_rank[dest_rank, 0].item())
                        duplicate_start = int(dup_counts_by_rank[dest_rank, 1].item())
                        dup_groups_by_rank[dest_rank, group_index] = (
                            dup_groups_by_rank.new_tensor(
                                (
                                    group_loffs[0],
                                    duplicate_start,
                                    duplicate_count,
                                )
                            )
                        )
                        dup_loffs_by_rank[
                            dest_rank,
                            duplicate_start : duplicate_start + duplicate_count,
                        ] = dup_loffs_by_rank.new_tensor(group_loffs[1:])
                        dup_counts_by_rank[dest_rank, 0] += 1
                        dup_counts_by_rank[dest_rank, 1] += duplicate_count
        canonicalize_local_dst(dst)
    elif dedup_mode == DEDUP_MODE_ZERO:
        canonicalize_local_dst(dst)
    else:
        assert src_cpu is not None
        canonicalize_local_dst(dst)
        entries_by_key = {}
        for local_offset in range(NvS):
            src_value = int(src_cpu[local_offset].item())
            if src_value < 0:
                continue
            source_rank = src_value // NvS
            offv = src_value - source_rank * NvS
            token = offv // K
            kidx = offv - token * K
            key = source_rank * S + token
            entries = entries_by_key.setdefault(key, {})
            if kidx in entries:
                raise AssertionError("torch planning reference: duplicate src kidx")
            entries[kidx] = local_offset

        groups = []
        for entries in entries_by_key.values():
            ordered = sorted(entries.items())
            if len(ordered) > 1:
                primary_loff = ordered[0][1]
                duplicate_loffs = [item[1] for item in ordered[1:]]
                groups.append((primary_loff, duplicate_loffs))
        groups.sort(key=lambda item: item[0])

        duplicate_start = 0
        for group_index, (primary_loff, duplicate_loffs) in enumerate(groups):
            duplicate_count = len(duplicate_loffs)
            dup_groups_by_rank[rank, group_index] = dup_groups_by_rank.new_tensor(
                (primary_loff, duplicate_start, duplicate_count)
            )
            dup_loffs_by_rank[
                rank,
                duplicate_start : duplicate_start + duplicate_count,
            ] = dup_loffs_by_rank.new_tensor(duplicate_loffs)
            duplicate_start += duplicate_count
        dup_counts_by_rank[rank, 0] = len(groups)
        dup_counts_by_rank[rank, 1] = duplicate_start

    remote_stats = remote_stats_all[rank]

    dev = tokens_per_expert.device
    ref_dedup_plan = ReferenceDedupPlan(
        dup_groups=dup_groups_by_rank[rank].to(dev),
        dup_loffs=dup_loffs_by_rank[rank].to(dev),
        dup_counts=dup_counts_by_rank[rank].to(dev),
    )
    plan = MoonEPCommPlan(
        dst=dst.to(dev),
        experts_to_copy=experts_to_copy.to(dev),
        zero_fill_ranges=zero_fill_by_rank[rank].to(dev),
        remote_stats=remote_stats.to(dev),
        N=N,
        R=R,
        E=E,
        B=B,
        NvS=NvS,
        K=K,
        dup_groups=ref_dedup_plan.dup_groups,
        dup_loffs=ref_dedup_plan.dup_loffs,
        dup_counts=ref_dedup_plan.dup_counts,
    )
    src = src_cpu.to(dev) if src_cpu is not None else None
    return cu_seqlens[rank].to(dev), plan, src


def gather_tensor(tensor: torch.Tensor, R: int) -> torch.Tensor:
    """Gather one same-shaped NPU tensor from every physical rank."""
    flat = tensor.contiguous().view(-1)
    gathered = torch.empty(R * flat.numel(), dtype=flat.dtype, device=flat.device)
    dist.all_gather_into_tensor(gathered, flat)
    return gathered.view(R, *tensor.shape)


def assert_all_ranks(
    ok: bool,
    rank: int,
    R: int,
    label: str,
    detail: str = "",
) -> None:
    """Fail every process when a condition fails on any physical rank."""
    local = torch.tensor([int(ok)], dtype=torch.int32, device="npu")
    all_ok = gather_tensor(local, R).cpu()
    if int(all_ok.sum().item()) != R:
        if not ok and detail:
            raise AssertionError(f"{label} failed on rank {rank}: {detail}")
        raise AssertionError(f"{label} failed on another rank")


def assert_tensor_equal_all_ranks(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rank: int,
    R: int,
    max_print: int = 5,
) -> None:
    """Compare tensor contracts and values, propagating failures to every rank."""
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    same_contract = actual.dtype == expected.dtype and tuple(actual.shape) == tuple(
        expected.shape
    )
    ok = same_contract and torch.equal(actual_cpu, expected_cpu)
    detail = ""
    if not same_contract:
        detail = (
            f"actual={actual.dtype} {tuple(actual.shape)}, "
            f"expected={expected.dtype} {tuple(expected.shape)}"
        )
    elif not ok:
        mask = actual_cpu != expected_cpu
        positions = mask.nonzero()[:max_print]
        lines = [f"{int(mask.sum())}/{actual_cpu.numel()} elements differ"]
        for position in positions:
            index = tuple(int(value) for value in position.tolist())
            lines.append(
                f"{name}{index}: actual={actual_cpu[index].item()} "
                f"expected={expected_cpu[index].item()}"
            )
        detail = "; ".join(lines)
    assert_all_ranks(ok, rank, R, name, detail)


def assert_planning_outputs_equal_all_ranks(
    actual_cu_seqlens: torch.Tensor,
    actual_plan,
    actual_src,
    expected_cu_seqlens: torch.Tensor,
    expected_plan,
    expected_src,
    rank: int,
    R: int,
    *,
    dedup_mode: str = DEDUP_MODE_CURRENT,
    label_prefix: str = "",
) -> None:
    """Compare both public return objects and every required plan field."""
    dedup_mode = normalize_dedup_mode(dedup_mode)
    assert_tensor_equal_all_ranks(
        f"{label_prefix}cu_seqlens",
        actual_cu_seqlens,
        expected_cu_seqlens,
        rank,
        R,
    )
    for field in (
        "dst",
        "experts_to_copy",
        "zero_fill_ranges",
        "remote_stats",
    ):
        assert_tensor_equal_all_ranks(
            f"{label_prefix}{field}",
            getattr(actual_plan, field),
            getattr(expected_plan, field),
            rank,
            R,
        )
    if dedup_mode == DEDUP_MODE_ZERO:
        for field in DEDUP_PLAN_FIELDS:
            assert_tensor_equal_all_ranks(
                f"{label_prefix}{field}",
                getattr(actual_plan, field),
                getattr(expected_plan, field),
                rank,
                R,
            )
    else:
        errors = dedup_plan_semantic_errors(
            f"{label_prefix}dedup plan", actual_plan, expected_plan
        )
        assert_all_ranks(
            not errors,
            rank,
            R,
            f"{label_prefix}dedup plan",
            "; ".join(errors[:5]),
        )

    if dedup_mode == DEDUP_MODE_SRC:
        src_contract = isinstance(actual_src, torch.Tensor) and isinstance(
            expected_src, torch.Tensor
        )
        assert_all_ranks(
            src_contract,
            rank,
            R,
            f"{label_prefix}src contract",
            f"actual={type(actual_src).__name__} "
            f"expected={type(expected_src).__name__}",
        )
        if src_contract:
            assert_tensor_equal_all_ranks(
                f"{label_prefix}src", actual_src, expected_src, rank, R
            )
    else:
        assert_all_ranks(
            actual_src is None and expected_src is None,
            rank,
            R,
            f"{label_prefix}src must be None",
            f"actual={type(actual_src).__name__} "
            f"expected={type(expected_src).__name__}",
        )


DEDUP_PLAN_FIELDS = (
    "dup_groups",
    "dup_loffs",
    "dup_counts",
)


def _dedup_group_map(plan, *, max_print=5):
    """Return order-independent dedup groups and internal consistency errors."""
    groups = plan.dup_groups.cpu()
    dup_loffs = plan.dup_loffs.cpu()
    counts = plan.dup_counts.cpu()
    NvS = dup_loffs.numel()
    errors = []
    mapping = {}

    group_count = int(counts[0].item())
    dup_count_total = int(counts[1].item())
    if group_count < 0 or group_count > NvS:
        errors.append(f"dup group count {group_count} out of range [0, {NvS}]")
        group_count = max(0, min(group_count, NvS))
    if dup_count_total < 0 or dup_count_total > NvS:
        errors.append(f"dup loff count {dup_count_total} out of range [0, {NvS}]")
        dup_count_total = max(0, min(dup_count_total, NvS))

    seen_dups = []
    for group_idx in range(group_count):
        primary, dup_start, dup_count = (
            int(value) for value in groups[group_idx].tolist()
        )
        if not (0 <= primary < NvS):
            errors.append(f"dup group {group_idx} primary {primary} out of range")
            continue
        if not (0 <= dup_start <= dup_count_total):
            errors.append(f"dup group {group_idx} dup_start {dup_start} out of range")
            continue
        if dup_count <= 0 or dup_start + dup_count > dup_count_total:
            errors.append(
                f"dup group {group_idx} invalid dup range "
                f"start={dup_start} count={dup_count} total={dup_count_total}"
            )
            continue
        duplicates = []
        for offset_value in dup_loffs[dup_start : dup_start + dup_count].tolist():
            offset = int(offset_value)
            if not (0 <= offset < NvS):
                errors.append(
                    f"dup group {group_idx} contains out-of-range dup loff " f"{offset}"
                )
                continue
            duplicates.append(offset)
            seen_dups.append(offset)
        if primary in mapping:
            errors.append(f"primary loff {primary} appears in multiple dup groups")
        if primary in duplicates:
            errors.append(f"dup group {group_idx} lists its own primary {primary}")
        mapping[primary] = tuple(sorted(duplicates))

    seen_dup_set = set(seen_dups)
    if len(seen_dups) != len(seen_dup_set):
        errors.append("dup_loffs contains repeated duplicate rows")
    if len(seen_dups) != dup_count_total:
        errors.append(
            f"dup loff count mismatch: header={dup_count_total}, "
            f"used={len(seen_dups)}"
        )
    overlap = seen_dup_set & set(mapping)
    if overlap:
        errors.append(
            "rows appear both as primary and duplicate: "
            f"{sorted(overlap)[:max_print]}"
        )
    return mapping, errors


def dedup_plan_semantic_errors(name, actual, expected, max_print=5):
    """Compare compact dedup plans without assuming group allocation order."""
    errors = []
    for field in DEDUP_PLAN_FIELDS:
        actual_tensor = getattr(actual, field)
        expected_tensor = getattr(expected, field)
        if actual_tensor.dtype != expected_tensor.dtype:
            errors.append(
                f"{field} dtype actual={actual_tensor.dtype} "
                f"expected={expected_tensor.dtype}"
            )
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            errors.append(
                f"{field} shape actual={tuple(actual_tensor.shape)} "
                f"expected={tuple(expected_tensor.shape)}"
            )
        if errors:
            return errors

    actual_counts = actual.dup_counts.cpu()
    expected_counts = expected.dup_counts.cpu()
    for index, label in ((0, "dup group count"), (1, "dup loff count")):
        actual_count = int(actual_counts[index].item())
        expected_count = int(expected_counts[index].item())
        if actual_count != expected_count:
            errors.append(
                f"{name} {label} actual={actual_count} expected={expected_count}"
            )

    actual_map, actual_errors = _dedup_group_map(actual, max_print=max_print)
    expected_map, expected_errors = _dedup_group_map(expected, max_print=max_print)
    errors.extend(f"{name} actual {error}" for error in actual_errors[:max_print])
    errors.extend(f"{name} expected {error}" for error in expected_errors[:max_print])
    if actual_map != expected_map:
        actual_keys = set(actual_map)
        expected_keys = set(expected_map)
        missing = sorted(expected_keys - actual_keys)[:max_print]
        extra = sorted(actual_keys - expected_keys)[:max_print]
        mismatched = [
            key
            for key in sorted(actual_keys & expected_keys)
            if actual_map[key] != expected_map[key]
        ][:max_print]
        errors.append(
            f"{name} duplicate group map differs: "
            f"missing={missing} extra={extra} mismatched={mismatched}"
        )
    return errors


def minimal_context_errors(ctx: dict) -> list[str]:
    """Return violations of the test-facing minimal Planning context contract."""
    required = {
        "rank",
        "R",
        "group",
        "S",
        "K",
        "E",
        "B",
        "N",
        "NvS_capacity",
        "NvS",
        "token_padding",
        "num_sms",
    }
    errors = [f"missing ctx field {key}" for key in sorted(required - ctx.keys())]
    errors.extend(
        f"unexpected ctx field {key}" for key in sorted(ctx.keys() - required)
    )
    return errors


def src_invariant_errors(ctx: dict, src) -> list[str]:
    """Check the rank-local MoonEP src_info tensor contract and encoding."""
    errors = []
    R = int(ctx["R"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    N = S * K
    if not isinstance(src, torch.Tensor):
        return ["src must be a tensor in src mode"]
    if src.dtype != torch.int32:
        errors.append(f"src must be int32, got {src.dtype}")
    if tuple(src.shape) != (NvS,):
        errors.append(f"src must have shape ({NvS},), got {tuple(src.shape)}")
    if not src.is_contiguous():
        errors.append("src must be contiguous")
    if src.device.type != "npu":
        errors.append("src must be on NPU")
    if errors:
        return errors

    src_cpu = src.cpu().to(torch.int64)
    valid = src_cpu >= 0
    if bool((src_cpu < -1).any()):
        errors.append("src contains a value below the -1 empty sentinel")
    if bool(valid.any()):
        values = src_cpu[valid]
        source_ranks = torch.div(values, NvS, rounding_mode="floor")
        source_offsets = values % NvS
        if bool(((source_ranks < 0) | (source_ranks >= R)).any()):
            errors.append("src contains an out-of-range source rank")
        if bool((source_offsets >= N).any()):
            errors.append("src contains a source offset outside [0, S*K)")
        if int(torch.unique(values).numel()) != int(values.numel()):
            errors.append("src contains repeated source provenance values")
    return errors


def planning_invariant_errors(
    case: KernelCase,
    ctx: dict,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    plan,
    cu_seqlens: torch.Tensor,
    src=None,
    *,
    dedup_mode: str = DEDUP_MODE_CURRENT,
) -> list[str]:
    """Check Planning and dedup results without CUDA/VMM layout state."""
    dedup_mode = normalize_dedup_mode(dedup_mode)
    errors: list[str] = []
    rank = int(ctx["rank"])
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    N = case.S * case.K

    if type(plan).__name__ != MoonEPCommPlan.__name__:
        errors.append(f"plan must be MoonEPCommPlan, got {type(plan).__name__}")

    topk_ok = (
        isinstance(topk_experts, torch.Tensor)
        and topk_experts.dtype == torch.int32
        and tuple(topk_experts.shape) == (case.S, case.K)
        and topk_experts.is_contiguous()
        and topk_experts.device.type == "npu"
    )
    if not topk_ok:
        errors.append("topk_experts must be contiguous int32 [S, K] on NPU")

    tpe_ok = (
        isinstance(tokens_per_expert, torch.Tensor)
        and tokens_per_expert.dtype == torch.int32
        and tuple(tokens_per_expert.shape) == (E,)
        and tokens_per_expert.is_contiguous()
        and tokens_per_expert.device.type == "npu"
    )
    if not tpe_ok:
        errors.append("tokens_per_expert must be contiguous int32 [E] on NPU")

    expected_fields = {
        "dst": (N,),
        "experts_to_copy": (R, B),
        "zero_fill_ranges": (E + B, 2),
        "remote_stats": (2,),
        "dup_groups": (NvS, 3),
        "dup_loffs": (NvS,),
        "dup_counts": (2,),
    }
    plan_tensors = {}
    for field, shape in expected_fields.items():
        tensor = getattr(plan, field, None)
        plan_tensors[field] = tensor
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"plan.{field} is not a tensor")
        elif tensor.dtype != torch.int32 or tuple(tensor.shape) != shape:
            errors.append(f"plan.{field} must be int32 {shape}")
        elif not tensor.is_contiguous():
            errors.append(f"plan.{field} must be contiguous")
        elif tensor.device.type != "npu":
            errors.append(f"plan.{field} must be on NPU")

    cu_ok = (
        isinstance(cu_seqlens, torch.Tensor)
        and cu_seqlens.dtype == torch.int32
        and tuple(cu_seqlens.shape) == (E + B,)
        and cu_seqlens.is_contiguous()
        and cu_seqlens.device.type == "npu"
    )
    if not cu_ok:
        errors.append("cu_seqlens must be contiguous int32 [E + B] on NPU")

    plan_ok = all(
        isinstance(plan_tensors[field], torch.Tensor)
        and plan_tensors[field].dtype == torch.int32
        and tuple(plan_tensors[field].shape) == shape
        and plan_tensors[field].is_contiguous()
        and plan_tensors[field].device.type == "npu"
        for field, shape in expected_fields.items()
    )
    if dedup_mode == DEDUP_MODE_SRC:
        src_errors = src_invariant_errors(ctx, src)
        errors.extend(src_errors)
        src_ok = not src_errors
    else:
        src_ok = src is None
        if not src_ok:
            errors.append(f"src must be None in {dedup_mode} mode")

    local_contract_ok = topk_ok and tpe_ok and cu_ok and plan_ok and src_ok
    contract_flag = torch.tensor(
        [int(local_contract_ok)], dtype=torch.int32, device="npu"
    )
    all_contracts_ok = bool(gather_tensor(contract_flag, R).cpu().all())
    if not all_contracts_ok:
        return errors

    if dedup_mode == DEDUP_MODE_ZERO:
        for field in DEDUP_PLAN_FIELDS:
            if bool((getattr(plan, field) != 0).any().item()):
                errors.append(f"plan.{field} must remain all-zero in zero mode")
    else:
        _, dedup_errors = _dedup_group_map(plan)
        errors.extend(f"dedup plan {error}" for error in dedup_errors[:5])

    local_hist = torch.bincount(
        topk_experts.reshape(-1).to(torch.int64).cpu(), minlength=E
    ).to(torch.int32)
    if not torch.equal(local_hist, tokens_per_expert.cpu()):
        errors.append("tokens_per_expert does not match local topk histogram")
    if int(tokens_per_expert.sum().item()) != N:
        errors.append("local token/expert count is not conserved")

    encoded = plan.dst.cpu().to(torch.int64)
    raw = torch.where(encoded < 0, -encoded - 1, encoded)
    destination = torch.div(raw, NvS, rounding_mode="floor")
    local_offset = raw % NvS
    if not bool(((destination >= 0) & (destination < R)).all()):
        errors.append("dst contains an out-of-range destination")
    if not bool(((local_offset >= 0) & (local_offset < NvS)).all()):
        errors.append("dst contains an out-of-range local offset")
    for token in range(case.S):
        seen: set[int] = set()
        for k_index in range(case.K):
            index = token * case.K + k_index
            dest = int(destination[index].item())
            should_encode = dest in seen
            if should_encode != (int(encoded[index].item()) < 0):
                errors.append(f"token {token} k {k_index} duplicate sign is incorrect")
                break
            seen.add(dest)

    cu_cpu = cu_seqlens.cpu()
    previous = 0
    for group_idx, end_value in enumerate(cu_cpu.tolist()):
        end = int(end_value)
        length = end - previous
        if length < 0:
            errors.append(f"cu_seqlens decreases at group {group_idx}")
            break
        if length and length % case.token_padding:
            errors.append(f"group {group_idx} is not padding aligned")
            break
        previous = end
    if previous > NvS:
        errors.append(f"cu_seqlens total {previous} exceeds NvS={NvS}")

    zero_fill = plan.zero_fill_ranges.cpu()
    previous = 0
    for group_idx, end_value in enumerate(cu_cpu.tolist()):
        end = int(end_value)
        pad_start, pad_count = (int(value) for value in zero_fill[group_idx].tolist())
        if pad_count < 0:
            errors.append(f"zero-fill count is negative at group {group_idx}")
            break
        if pad_count == 0:
            if pad_start != 0:
                errors.append(f"zero-fill start must be zero at group {group_idx}")
                break
        elif not (
            previous <= pad_start < end
            and pad_start + pad_count == end
            and pad_count < case.token_padding
        ):
            errors.append(f"invalid zero-fill range at group {group_idx}")
            break
        previous = end

    experts = plan.experts_to_copy.cpu()
    if not bool(((experts == -1) | ((experts >= 0) & (experts < E))).all()):
        errors.append("experts_to_copy contains invalid experts")
    epn = E // R
    for destination_rank, row in enumerate(experts.tolist()):
        saw_unused = False
        for expert in row:
            if expert == -1:
                saw_unused = True
            elif saw_unused:
                errors.append(f"experts_to_copy row {destination_rank} has a hole")
                break
            elif expert // epn == destination_rank:
                errors.append(
                    f"experts_to_copy row {destination_rank} contains local expert"
                )
                break

    stats = plan.remote_stats.cpu()
    if bool((stats < 0).any()):
        errors.append("remote_stats contains a negative count")
    valid_local = int((experts[rank] >= 0).sum().item())
    if valid_local != min(B, int(stats[0].item())):
        errors.append("remote_stats[0] disagrees with experts_to_copy")
    outbound_prefetches = int(((experts >= 0) & (experts // epn == rank)).sum().item())
    if outbound_prefetches != int(stats[1].item()):
        errors.append("remote_stats[1] disagrees with experts_to_copy")

    all_topk = gather_tensor(topk_experts, R).cpu().reshape(-1)
    all_tpe = gather_tensor(tokens_per_expert, R).cpu().sum(dim=0)
    expected_hist = torch.bincount(all_topk.to(torch.int64), minlength=E).to(
        torch.int32
    )
    if not torch.equal(all_tpe, expected_hist):
        errors.append("global per-expert token count is not conserved")

    all_dst = gather_tensor(plan.dst, R).cpu().to(torch.int64).reshape(-1)
    all_raw = torch.where(all_dst < 0, -all_dst - 1, all_dst)
    all_destinations = torch.div(all_raw, NvS, rounding_mode="floor")
    if bool(((all_destinations < 0) | (all_destinations >= R)).any()):
        errors.append("global dst contains an out-of-range destination")
    else:
        destination_counts = torch.bincount(all_destinations, minlength=R)
        if bool((destination_counts > int(ctx["NvS_capacity"])).any()):
            errors.append("a destination exceeds its real-token capacity")

        all_cu = gather_tensor(cu_seqlens, R).cpu()
        all_local_offsets = all_raw % NvS
        for destination_rank in range(R):
            mask = all_destinations == destination_rank
            if not bool(mask.any()):
                continue
            offsets = all_local_offsets[mask]
            used_end = int(all_cu[destination_rank, -1].item())
            if bool((offsets >= used_end).any()):
                errors.append(
                    f"destination {destination_rank} has dst beyond cu_seqlens"
                )
            if int(torch.unique(offsets).numel()) != int(offsets.numel()):
                errors.append(f"destination {destination_rank} reuses a dst offset")

    if dedup_mode == DEDUP_MODE_SRC:
        all_src = gather_tensor(src, R).cpu().to(torch.int64)
        valid_src = all_src >= 0
        if int(valid_src.sum().item()) != R * N:
            errors.append(
                f"global src valid count {int(valid_src.sum().item())} "
                f"does not match routed entries {R * N}"
            )
        all_dst_matrix = all_dst.view(R, N)
        all_raw_matrix = torch.where(
            all_dst_matrix < 0, -all_dst_matrix - 1, all_dst_matrix
        )
        for source_rank in range(R):
            for offv in range(N):
                raw_dst = int(all_raw_matrix[source_rank, offv].item())
                dest_rank = raw_dst // NvS
                dest_loff = raw_dst % NvS
                expected_src = source_rank * NvS + offv
                actual_src = int(all_src[dest_rank, dest_loff].item())
                if actual_src != expected_src:
                    errors.append(
                        f"src/dst mismatch source=({source_rank},{offv}) "
                        f"dest=({dest_rank},{dest_loff}) "
                        f"actual={actual_src} expected={expected_src}"
                    )
                    break
            if errors and errors[-1].startswith("src/dst mismatch"):
                break

        dedup_map, dedup_errors = _dedup_group_map(plan)
        errors.extend(f"src dedup plan {error}" for error in dedup_errors[:5])
        local_src = src.cpu().to(torch.int64)
        for primary, duplicates in dedup_map.items():
            offsets = (primary, *duplicates)
            decoded = []
            for local_offset in offsets:
                value = int(local_src[local_offset].item())
                if value < 0:
                    errors.append(f"dedup loff {local_offset} points to empty src slot")
                    continue
                source_rank = value // NvS
                offv = value % NvS
                decoded.append((source_rank, offv // K, offv % K))
            if not decoded:
                continue
            keys = {(item[0], item[1]) for item in decoded}
            if len(keys) != 1:
                errors.append(f"dedup group primary {primary} spans source token keys")
            elif decoded[0][2] != min(item[2] for item in decoded):
                errors.append(f"dedup group primary {primary} is not the lowest kidx")
    return errors


def assert_all_ranks_compute(ctx: dict, rank: int, R: int) -> None:
    """Verify that every physical rank executed its Planning implementation."""
    count = torch.tensor(
        [int(ctx.get("_planning_launch_count", -1))],
        dtype=torch.int32,
        device="npu",
    )
    counts = gather_tensor(count, R).cpu().reshape(-1)
    ok = all(int(value.item()) > 0 for value in counts)
    assert_all_ranks(
        ok,
        rank,
        R,
        "all ranks compute",
        f"launch counts={counts.tolist()}",
    )


def _align_up(value: int, alignment: int) -> int:
    """Round an integer up for coverage calculations."""
    return ((value + alignment - 1) // alignment) * alignment


def assert_planning_step1_case_coverage() -> None:
    """Preserve the official step-1 boundary coverage assertions."""
    params = []
    for R in (2, 4):
        for case in PLANNING_CASES:
            if R < case.min_R or (case.max_R is not None and R > case.max_R):
                continue
            E = case.E(R)
            segment = (E + case.num_sms - 1) // case.num_sms
            experts_per_block = _align_up(segment, 32)
            s1_cols = min(experts_per_block, 512)
            params.append(
                {
                    "E": E,
                    "experts_per_block": experts_per_block,
                    "s1_cols": s1_cols,
                    "work_ctas": (E + experts_per_block - 1) // experts_per_block,
                    "group_spans_ctas": experts_per_block < case.epn,
                    "has_segment_tail": (
                        experts_per_block > s1_cols and experts_per_block % s1_cols != 0
                    ),
                }
            )
    assert any(item["E"] > 2048 for item in params)
    assert any(item["s1_cols"] > 32 for item in params)
    assert any(item["s1_cols"] == 512 for item in params)
    assert any(item["experts_per_block"] > item["s1_cols"] for item in params)
    assert any(item["has_segment_tail"] for item in params)
    assert any(item["work_ctas"] > 1 and item["group_spans_ctas"] for item in params)


@pytest.fixture(scope="session")
def dist_env():
    """Initialize a real multi-rank HCCL session and bind LOCAL_RANK."""
    if "RANK" not in os.environ:
        pytest.fail("distributed Planning tests must be launched with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_ids = [
        int(value) for value in os.environ.get("MOONEP_NPU_IDS", "2,3").split(",")
    ]
    if len(device_ids) < 2 or len(set(device_ids)) != len(device_ids):
        pytest.fail(
            "MOONEP_NPU_IDS must name at least two unique NPUs: " f"{device_ids}"
        )
    if local_rank >= len(device_ids):
        pytest.fail(f"LOCAL_RANK={local_rank} has no corresponding NPU in {device_ids}")
    device = npu_index_for_local_rank(local_rank, device_ids)
    trace = os.environ.get("MOONEP_PLANNING_TRACE") == "1"
    if trace:
        print(
            f"[fixture local_rank={local_rank}] set_device({device}) begin",
            flush=True,
        )
    torch.npu.set_device(device)
    if trace:
        print(f"[fixture local_rank={local_rank}] init_process_group begin", flush=True)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if trace:
        print(
            f"[fixture rank={rank}] init_process_group done R={world_size}", flush=True
        )
    if world_size < 2:
        pytest.fail(
            f"Planning tests require at least two real ranks, got R={world_size}"
        )
    if world_size != len(device_ids):
        pytest.fail(
            f"world size {world_size} does not match MOONEP_NPU_IDS {device_ids}"
        )
    yield rank, world_size
    if dist.is_initialized():
        dist.destroy_process_group()
