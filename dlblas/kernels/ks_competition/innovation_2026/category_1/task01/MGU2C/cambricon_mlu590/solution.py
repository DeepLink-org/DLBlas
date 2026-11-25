"""Triton implementation of grouped expert routing.

The forward path deliberately has no PyTorch computation fallback.  PyTorch is
used only for output allocation and module/interface compatibility; all scores,
group selection, expert selection, and normalization are computed by Triton.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _target_device():
    """Prefer the MLU device for this profile, with a CUDA smoke-test fallback."""
    mlu = getattr(torch, "mlu", None)
    if mlu is not None and mlu.is_available():
        return torch.device("mlu")
    return torch.device("cuda")


# The MLU590 static reduction default has 83 independent 256-lane owners.  That is
# enough program-level parallelism for the architecture-level estimate of 32
# compute cores, while W1 avoids unverified cross-core task overhead. Cambricon
# ``num_warps`` selects BLOCK/U1/Ux geometry rather than CUDA-sized warps.
_CHIP_CONFIGS = {
    "auto": (4,),
    "ascend_a2": (2,),
    "metax_c500": (1,),
    "iluvatar_bi150": (4,),
    "enflame_s60": (1,),
    "thead_810e": (2,),
    "cambricon_mlu590": (1,),
    "hygon_bw1000": (4,),
    "mthreads_ph100": (4,),
    "kunlun_p800": (2,),
    "biren_br106m": (4,),
}


# MLU launch metadata is deliberately a closed set.  In the Cambricon backend
# ``num_warps`` controls the task geometry rather than CUDA warp*32 threads;
# W1 is therefore the conservative default while W2/W4 and S1/S2/S3 remain
# reproducible target-side A/B candidates.  The official specialization uses
# only Triton-3.2-era primitives already exercised by the generic kernel.
_MLU590_CONFIGS = {
    "generic_w1s1": ("generic", 1, 1),
    "static_reduceid_w1s1": ("static_reduceid", 1, 1),
    "static_reduceid_w2s2": ("static_reduceid", 2, 2),
    "static_full_w1s1": ("static_full", 1, 1),
    "static_full_w1s2": ("static_full", 1, 2),
    "static_full_w1s3": ("static_full", 1, 3),
    "static_full_w2s1": ("static_full", 2, 1),
    "static_full_w2s2": ("static_full", 2, 2),
    "static_full_w2s3": ("static_full", 2, 3),
    "static_full_w4s1": ("static_full", 4, 1),
    "static_full_w4s2": ("static_full", 4, 2),
    "static_full_w4s3": ("static_full", 4, 3),
    "static_compact_w1s1": ("static_compact", 1, 1),
    "static_compact_w1s2": ("static_compact", 1, 2),
    "static_compact_w1s3": ("static_compact", 1, 3),
    "static_compact_w2s1": ("static_compact", 2, 1),
    "static_compact_w2s2": ("static_compact", 2, 2),
    "static_compact_w2s3": ("static_compact", 2, 3),
    "static_compact_w4s1": ("static_compact", 4, 1),
    "static_compact_w4s2": ("static_compact", 4, 2),
    "static_compact_w4s3": ("static_compact", 4, 3),
}


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
    config = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])
    if chip == "cambricon_mlu590":
        requested = (
            os.environ.get("KERNELSWIFT_TASK01_CONFIG", "static_reduceid_w1s1")
            .strip()
            .lower()
        )
        # Preserve the earlier WARPS knob when CONFIG is not explicitly set.
        if "KERNELSWIFT_TASK01_CONFIG" not in os.environ:
            legacy_warps = os.environ.get("KERNELSWIFT_TASK01_WARPS", "").strip()
            if legacy_warps in ("1", "2", "4"):
                requested = f"static_full_w{legacy_warps}s1"
        return _MLU590_CONFIGS.get(requested, _MLU590_CONFIGS["static_reduceid_w1s1"])
    # Small, explicit A/B surface for the fixed 256-lane reduction.  Keeping
    # this lookup in ``__init__`` avoids environment access in the timed path.
    # Invalid values deliberately retain the chip profile's safe default.
    if chip in ("ascend_a2", "metax_c500", "enflame_s60", "cambricon_mlu590"):
        variants = {
            "1": (1,),
            "4": (4,),
        }
        variant = os.environ.get("KERNELSWIFT_TASK01_WARPS", "").strip()
        config = variants.get(variant, config)
    return "generic", config[0], 1


def _get_persistent_programs():
    """Return an opt-in bounded token-owner grid.

    Static-reduceid/83 programs is the MLU590 default. These generic persistent
    forms preserve identical math and exist only for target-side timing; no
    device property query or timed-path environment access is used.
    """
    chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
    if chip not in ("enflame_s60", "cambricon_mlu590"):
        return 0
    if chip == "enflame_s60":
        modes = {"persistent_p24": 24, "persistent_p48": 48}
    else:
        modes = {
            "persistent_p16": 16,
            "persistent_p32": 32,
            "persistent_p64": 64,
        }
    mode = os.environ.get("KERNELSWIFT_TASK01_MODE", "direct").strip().lower()
    return modes.get(mode, 0)


@triton.jit
def _grouped_topk_kernel(
    gating_output,
    topk_weights,
    topk_ids,
    stride_gating_token,
    stride_gating_expert,
    NUM_EXPERTS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_TOTAL: tl.constexpr,
    USE_SOFTMAX: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    ROUTED_SCALE: tl.constexpr,
):
    token = tl.program_id(0)

    group_lane = tl.arange(0, BLOCK_GROUPS)
    expert_lane = tl.arange(0, BLOCK_EXPERTS)
    group = group_lane[:, None]
    expert_in_group = expert_lane[None, :]
    expert_id = group * EXPERTS_PER_GROUP + expert_in_group
    valid = (
        (group < NUM_GROUPS)
        & (expert_in_group < EXPERTS_PER_GROUP)
        & (expert_id < NUM_EXPERTS)
    )

    logits = tl.load(
        gating_output + token * stride_gating_token + expert_id * stride_gating_expert,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    flat_logits = tl.reshape(logits, (BLOCK_TOTAL,))

    # Both supported score functions are strictly increasing.  Group and
    # expert Top-K therefore operate directly on logits; score evaluation is
    # needed only for the TOPK winners.
    group_logits = tl.reshape(flat_logits, (BLOCK_GROUPS, BLOCK_EXPERTS))
    group_scores = tl.max(group_logits, axis=1)

    # Select the winning groups.  Repeated argmax keeps the emitted order and
    # avoids materialising an O(G^2) rank matrix.
    chosen_groups = tl.zeros((BLOCK_GROUPS,), dtype=tl.int1)
    group_candidates = group_scores
    for _ in range(TOPK_GROUP):
        best_group = tl.argmax(group_candidates, axis=0, tie_break_left=True)
        is_best = group_lane == best_group
        chosen_groups = chosen_groups | is_best
        group_candidates = tl.where(is_best, -float("inf"), group_candidates)

    candidates = tl.reshape(
        tl.where(chosen_groups[:, None], group_logits, -float("inf")),
        (BLOCK_TOTAL,),
    )
    flat_lane = tl.arange(0, BLOCK_TOTAL)

    # For softmax+renormalize, the global softmax denominator cancels exactly;
    # only TOPK exponentials remain.  Without renormalization the full softmax
    # denominator is still required.  Subtracting the global maximum is stable
    # and valid because the globally best expert's group is always selected.
    row_max = 0.0
    full_softmax_sum = 1.0
    if USE_SOFTMAX:
        row_max = tl.max(flat_logits, axis=0)
        if not RENORMALIZE:
            full_softmax_sum = tl.sum(tl.exp(flat_logits - row_max), axis=0)

    # Select once and keep the ordered winners in an eight-lane SSA vector.
    # The former renormalized path repeated the same eight 256-way argmax
    # reductions solely to recover the output order.
    rank_lane = tl.arange(0, TOPK)
    selected_logits = tl.full((TOPK,), -float("inf"), tl.float32)
    selected_ids = tl.zeros((TOPK,), tl.int32)
    tmp = candidates
    for rank in range(TOPK):
        best = tl.argmax(tmp, axis=0, tie_break_left=True)
        best_logit = tl.max(tmp, axis=0)
        is_rank = rank_lane == rank
        selected_logits = tl.where(is_rank, best_logit, selected_logits)
        selected_ids = tl.where(is_rank, best, selected_ids)
        tmp = tl.where(flat_lane == best, -float("inf"), tmp)

    if USE_SOFTMAX:
        selected_values = tl.exp(selected_logits - row_max)
    else:
        selected_values = 1.0 / (1.0 + tl.exp(-selected_logits))
    if RENORMALIZE:
        denominator = tl.sum(selected_values, axis=0)
        selected_values = selected_values / denominator
    elif USE_SOFTMAX:
        selected_values = selected_values / full_softmax_sum

    best_group = selected_ids // BLOCK_EXPERTS
    best_local = selected_ids - best_group * BLOCK_EXPERTS
    original_expert = best_group * EXPERTS_PER_GROUP + best_local
    tl.store(
        topk_weights + token * TOPK + rank_lane,
        selected_values * ROUTED_SCALE,
    )
    tl.store(topk_ids + token * TOPK + rank_lane, original_expert)


@triton.jit
def _grouped_topk_official_kernel(
    gating_output,
    topk_weights,
    topk_ids,
    COMPACT_SELECTED: tl.constexpr,
):
    """Static T=83/E=256/G=8/Kg=4/K=8 softmax+renorm kernel.

    ``COMPACT_SELECTED=False`` is the safe default: it retains the full expert
    order and reads every logit exactly once.  The opt-in compact specialization
    sorts the four chosen groups by original group id before reloading their
    128 experts.  That halves the width of all expert Top-K reductions without
    changing this kernel's left-most tie rule, at the cost of 50% more logit
    traffic.  Both branches are compile-time static.
    """
    token = tl.program_id(0)
    group_lane = tl.arange(0, 8)
    expert_lane = tl.arange(0, 32)
    expert_id = group_lane[:, None] * 32 + expert_lane[None, :]
    group_logits = tl.load(gating_output + token * 256 + expert_id).to(tl.float32)
    group_scores = tl.max(group_logits, axis=1)

    chosen_groups = tl.zeros((8,), dtype=tl.int1)
    group_candidates = group_scores
    for _ in range(4):
        best_group = tl.argmax(group_candidates, axis=0, tie_break_left=True)
        is_best = group_lane == best_group
        chosen_groups = chosen_groups | is_best
        group_candidates = tl.where(is_best, -float("inf"), group_candidates)

    if COMPACT_SELECTED:
        # Reorder by original group id, not by group score.  Concatenating the
        # four groups in that order makes compact lane order identical to the
        # corresponding subsequence of the full 0..255 expert-id order.
        group_rank_lane = tl.arange(0, 4)
        ordered_groups = tl.zeros((4,), dtype=tl.int32)
        remaining_groups = chosen_groups
        for rank in range(4):
            inverse_id = tl.where(
                remaining_groups,
                (7 - group_lane).to(tl.float32),
                -float("inf"),
            )
            next_group = tl.argmax(inverse_id, axis=0, tie_break_left=True)
            ordered_groups = tl.where(
                group_rank_lane == rank, next_group, ordered_groups
            )
            remaining_groups = remaining_groups & (group_lane != next_group)

        compact_expert_id = ordered_groups[:, None] * 32 + expert_lane[None, :]
        compact_logits = tl.load(gating_output + token * 256 + compact_expert_id).to(
            tl.float32
        )
        candidates = tl.reshape(compact_logits, (128,))
        candidate_lane = tl.arange(0, 128)
    else:
        candidates = tl.reshape(
            tl.where(
                chosen_groups[:, None],
                group_logits,
                -float("inf"),
            ),
            (256,),
        )
        candidate_lane = tl.arange(0, 256)

    rank_lane = tl.arange(0, 8)
    selected_logits = tl.full((8,), -float("inf"), tl.float32)
    selected_ids = tl.zeros((8,), tl.int32)
    tmp = candidates
    for rank in range(8):
        best = tl.argmax(tmp, axis=0, tie_break_left=True)
        best_logit = tl.max(tmp, axis=0)
        is_rank = rank_lane == rank
        selected_logits = tl.where(is_rank, best_logit, selected_logits)
        selected_ids = tl.where(is_rank, best, selected_ids)
        tmp = tl.where(candidate_lane == best, -float("inf"), tmp)

    # Let z_i=exp(x_i)/D be the reference softmax score.  Renormalizing the
    # eight winners gives z_i/sum(z_j)=exp(x_i-m)/sum(exp(x_j-m)); D cancels.
    # ``m`` can be the maximum of only the eight winners, saving the generic
    # path's separate 256-lane row-max reduction with no arithmetic change.
    selected_max = tl.max(selected_logits, axis=0)
    selected_values = tl.exp(selected_logits - selected_max)
    selected_values = selected_values / tl.sum(selected_values, axis=0)

    if COMPACT_SELECTED:
        selected_group_slot = selected_ids // 32
        selected_local_id = selected_ids - selected_group_slot * 32
        selected_group_id = tl.sum(
            tl.where(
                selected_group_slot[:, None] == group_rank_lane[None, :],
                ordered_groups[None, :],
                0,
            ),
            axis=1,
        )
        output_ids = selected_group_id * 32 + selected_local_id
    else:
        output_ids = selected_ids

    tl.store(topk_weights + token * 8 + rank_lane, selected_values)
    tl.store(topk_ids + token * 8 + rank_lane, output_ids)


@triton.jit
def _grouped_topk_reduceid_official_kernel(
    gating_output,
    topk_weights,
    topk_ids,
):
    """Conservative fixed-shape path without ``tl.argmax`` or ``tl.int1``.

    A float maximum first identifies the winning value.  A second float
    maximum over ``N - 1 - lane`` among equal winners recovers the lowest lane
    id exactly for N<=256.  This uses only reductions already observed in the
    successful MLU Task08 program, while preserving the established left-tie
    policy.  It remains opt-in pending target compilation and timing.
    """
    token = tl.program_id(0)
    group_lane = tl.arange(0, 8)
    expert_lane = tl.arange(0, 32)
    expert_id = group_lane[:, None] * 32 + expert_lane[None, :]
    group_logits = tl.load(gating_output + token * 256 + expert_id).to(tl.float32)
    group_scores = tl.max(group_logits, axis=1)

    chosen_groups = tl.zeros((8,), dtype=tl.float32)
    group_candidates = group_scores
    for _ in range(4):
        best_group_score = tl.max(group_candidates, axis=0)
        reverse_group_id = tl.where(
            group_candidates == best_group_score,
            (7 - group_lane).to(tl.float32),
            -float("inf"),
        )
        best_reverse_group_id = tl.max(reverse_group_id, axis=0)
        best_group = (7.0 - best_reverse_group_id).to(tl.int32)
        is_best = group_lane == best_group
        chosen_groups = chosen_groups + tl.where(is_best, 1.0, 0.0)
        group_candidates = tl.where(is_best, -float("inf"), group_candidates)

    candidates = tl.reshape(
        tl.where(
            chosen_groups[:, None] != 0,
            group_logits,
            -float("inf"),
        ),
        (256,),
    )
    candidate_lane = tl.arange(0, 256)
    rank_lane = tl.arange(0, 8)
    selected_logits = tl.full((8,), -float("inf"), tl.float32)
    selected_ids = tl.zeros((8,), tl.int32)
    tmp = candidates
    for rank in range(8):
        best_logit = tl.max(tmp, axis=0)
        reverse_expert_id = tl.where(
            tmp == best_logit,
            (255 - candidate_lane).to(tl.float32),
            -float("inf"),
        )
        best_reverse_expert_id = tl.max(reverse_expert_id, axis=0)
        best_expert = (255.0 - best_reverse_expert_id).to(tl.int32)
        is_rank = rank_lane == rank
        selected_logits = tl.where(is_rank, best_logit, selected_logits)
        selected_ids = tl.where(is_rank, best_expert, selected_ids)
        tmp = tl.where(candidate_lane == best_expert, -float("inf"), tmp)

    selected_max = tl.max(selected_logits, axis=0)
    selected_values = tl.exp(selected_logits - selected_max)
    selected_values = selected_values / tl.sum(selected_values, axis=0)
    tl.store(topk_weights + token * 8 + rank_lane, selected_values)
    tl.store(topk_ids + token * 8 + rank_lane, selected_ids)


@triton.jit
def _grouped_topk_persistent_kernel(
    gating_output,
    topk_weights,
    topk_ids,
    stride_gating_token,
    stride_gating_expert,
    NUM_TOKENS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_TOTAL: tl.constexpr,
    USE_SOFTMAX: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    ROUTED_SCALE: tl.constexpr,
):
    """Bounded token owner with the same reductions as the direct kernel."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group_lane = tl.arange(0, BLOCK_GROUPS)
    expert_lane = tl.arange(0, BLOCK_EXPERTS)
    group = group_lane[:, None]
    expert_in_group = expert_lane[None, :]
    expert_id = group * EXPERTS_PER_GROUP + expert_in_group
    valid = (
        (group < NUM_GROUPS)
        & (expert_in_group < EXPERTS_PER_GROUP)
        & (expert_id < NUM_EXPERTS)
    )
    flat_lane = tl.arange(0, BLOCK_TOTAL)
    rank_lane = tl.arange(0, TOPK)

    for token in range(pid, NUM_TOKENS, num_programs):
        logits = tl.load(
            gating_output
            + token * stride_gating_token
            + expert_id * stride_gating_expert,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        flat_logits = tl.reshape(logits, (BLOCK_TOTAL,))
        group_logits = tl.reshape(flat_logits, (BLOCK_GROUPS, BLOCK_EXPERTS))
        group_scores = tl.max(group_logits, axis=1)

        # Preserve the direct kernel's ordered repeated-argmax selection and
        # left-tie rule.  A compact 128-lane domain is intentionally not used:
        # ordering selected groups by score would change expert-id tie breaks.
        chosen_groups = tl.zeros((BLOCK_GROUPS,), dtype=tl.int1)
        group_candidates = group_scores
        for _ in range(TOPK_GROUP):
            best_group = tl.argmax(group_candidates, axis=0, tie_break_left=True)
            is_best = group_lane == best_group
            chosen_groups = chosen_groups | is_best
            group_candidates = tl.where(is_best, -float("inf"), group_candidates)

        candidates = tl.reshape(
            tl.where(chosen_groups[:, None], group_logits, -float("inf")),
            (BLOCK_TOTAL,),
        )
        row_max = 0.0
        full_softmax_sum = 1.0
        if USE_SOFTMAX:
            row_max = tl.max(flat_logits, axis=0)
            if not RENORMALIZE:
                full_softmax_sum = tl.sum(tl.exp(flat_logits - row_max), axis=0)

        selected_logits = tl.full((TOPK,), -float("inf"), tl.float32)
        selected_ids = tl.zeros((TOPK,), tl.int32)
        tmp = candidates
        for rank in range(TOPK):
            best = tl.argmax(tmp, axis=0, tie_break_left=True)
            best_logit = tl.max(tmp, axis=0)
            is_rank = rank_lane == rank
            selected_logits = tl.where(is_rank, best_logit, selected_logits)
            selected_ids = tl.where(is_rank, best, selected_ids)
            tmp = tl.where(flat_lane == best, -float("inf"), tmp)

        if USE_SOFTMAX:
            selected_values = tl.exp(selected_logits - row_max)
        else:
            selected_values = 1.0 / (1.0 + tl.exp(-selected_logits))
        if RENORMALIZE:
            denominator = tl.sum(selected_values, axis=0)
            selected_values = selected_values / denominator
        elif USE_SOFTMAX:
            selected_values = selected_values / full_softmax_sum

        best_group = selected_ids // BLOCK_EXPERTS
        best_local = selected_ids - best_group * BLOCK_EXPERTS
        original_expert = best_group * EXPERTS_PER_GROUP + best_local
        tl.store(
            topk_weights + token * TOPK + rank_lane,
            selected_values * ROUTED_SCALE,
        )
        tl.store(topk_ids + token * TOPK + rank_lane, original_expert)


class ModelNew(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        if scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(f"Unsupported scoring_func: {scoring_func}")
        self.topk = int(topk)
        self.renormalize = bool(renormalize)
        self.num_expert_group = int(num_expert_group)
        self.topk_group = int(topk_group)
        self.scoring_func = scoring_func
        self.routed_scaling_factor = float(routed_scaling_factor)
        (
            self._implementation,
            self._num_warps,
            self._num_stages,
        ) = _get_launch_config()
        self._persistent_programs = (
            _get_persistent_programs() if self._implementation == "generic" else 0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != gating_output.shape[0]:
            raise AssertionError("hidden_states and gating_output must have equal T")
        if gating_output.ndim != 2:
            raise ValueError("gating_output must have shape [T, E]")

        num_tokens, num_experts = gating_output.shape
        if num_experts % self.num_expert_group != 0:
            raise ValueError("num_experts must be divisible by num_expert_group")
        experts_per_group = num_experts // self.num_expert_group
        if not (0 < self.topk_group <= self.num_expert_group):
            raise ValueError("topk_group is outside the available group range")
        if not (0 < self.topk <= self.topk_group * experts_per_group):
            raise ValueError("topk exceeds the experts in the selected groups")

        weights = torch.empty(
            (num_tokens, self.topk), device=gating_output.device, dtype=torch.float32
        )
        ids = torch.empty(
            (num_tokens, self.topk), device=gating_output.device, dtype=torch.int32
        )

        block_groups = triton.next_power_of_2(self.num_expert_group)
        block_experts = triton.next_power_of_2(experts_per_group)
        block_total = block_groups * block_experts
        use_official_specialization = (
            self._implementation in ("static_full", "static_compact", "static_reduceid")
            and num_tokens == 83
            and num_experts == 256
            and gating_output.dtype == torch.float32
            and gating_output.is_contiguous()
            and self.num_expert_group == 8
            and self.topk_group == 4
            and self.topk == 8
            and self.scoring_func == "softmax"
            and self.renormalize
            and self.routed_scaling_factor == 1.0
        )
        if use_official_specialization and self._implementation == "static_reduceid":
            _grouped_topk_reduceid_official_kernel[(83,)](
                gating_output,
                weights,
                ids,
                num_warps=self._num_warps,
                num_stages=self._num_stages,
            )
        elif use_official_specialization:
            _grouped_topk_official_kernel[(83,)](
                gating_output,
                weights,
                ids,
                COMPACT_SELECTED=self._implementation == "static_compact",
                num_warps=self._num_warps,
                num_stages=self._num_stages,
            )
        elif self._persistent_programs:
            physical_programs = min(num_tokens, self._persistent_programs)
            _grouped_topk_persistent_kernel[(physical_programs,)](
                gating_output,
                weights,
                ids,
                gating_output.stride(0),
                gating_output.stride(1),
                NUM_TOKENS=num_tokens,
                NUM_EXPERTS=num_experts,
                NUM_GROUPS=self.num_expert_group,
                EXPERTS_PER_GROUP=experts_per_group,
                TOPK_GROUP=self.topk_group,
                TOPK=self.topk,
                BLOCK_GROUPS=block_groups,
                BLOCK_EXPERTS=block_experts,
                BLOCK_TOTAL=block_total,
                USE_SOFTMAX=self.scoring_func == "softmax",
                RENORMALIZE=self.renormalize,
                ROUTED_SCALE=self.routed_scaling_factor,
                num_warps=self._num_warps,
                num_stages=self._num_stages,
            )
        else:
            _grouped_topk_kernel[(num_tokens,)](
                gating_output,
                weights,
                ids,
                gating_output.stride(0),
                gating_output.stride(1),
                NUM_EXPERTS=num_experts,
                NUM_GROUPS=self.num_expert_group,
                EXPERTS_PER_GROUP=experts_per_group,
                TOPK_GROUP=self.topk_group,
                TOPK=self.topk,
                BLOCK_GROUPS=block_groups,
                BLOCK_EXPERTS=block_experts,
                BLOCK_TOTAL=block_total,
                USE_SOFTMAX=self.scoring_func == "softmax",
                RENORMALIZE=self.renormalize,
                ROUTED_SCALE=self.routed_scaling_factor,
                num_warps=self._num_warps,
                num_stages=self._num_stages,
            )
        return weights, ids


class Model(ModelNew):
    """Compatibility name for runners that still import ``Model``."""


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    device = _target_device()
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    return [hidden_states, gating_output]


def get_init_inputs():
    return [8, True, 8, 4]
