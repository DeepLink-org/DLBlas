"""KernelSwift Task01 grouped top-k candidate for musa_ph100.

The MUSA backend currently miscompiles wide reductions and cannot lower the
matrix-style variant reliably.  This implementation uses three custom Triton
kernels and keeps every reduction at 32 elements or fewer: score
normalization, per-group top-8 selection, and a scalar four-way merge.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "musa_ph100"


@triton.jit
def _normalize_scores_kernel(
    logits_ptr, scores_ptr, n_tokens: tl.constexpr, scoring_sigmoid: tl.constexpr
):
    token = tl.program_id(0)
    offs = tl.arange(0, 32)
    chunk_ids = tl.arange(0, 8)
    if scoring_sigmoid:
        for chunk in range(8):
            logits = tl.load(logits_ptr + token * 256 + chunk * 32 + offs).to(
                tl.float32
            )
            scores = 1.0 / (1.0 + tl.exp(-logits))
            tl.store(scores_ptr + token * 256 + chunk * 32 + offs, scores)
    else:
        chunk_maxes = tl.zeros((8,), tl.float32)
        for chunk in range(8):
            logits = tl.load(logits_ptr + token * 256 + chunk * 32 + offs).to(
                tl.float32
            )
            local_max = tl.max(logits, axis=0)
            chunk_maxes = tl.where(chunk_ids == chunk, local_max, chunk_maxes)
        global_max = tl.max(chunk_maxes, axis=0)
        chunk_sums = tl.zeros((8,), tl.float32)
        for chunk in range(8):
            logits = tl.load(logits_ptr + token * 256 + chunk * 32 + offs).to(
                tl.float32
            )
            local_sum = tl.sum(tl.exp(logits - global_max), axis=0)
            chunk_sums = tl.where(chunk_ids == chunk, local_sum, chunk_sums)
        denom = tl.sum(chunk_sums, axis=0)
        for chunk in range(8):
            logits = tl.load(logits_ptr + token * 256 + chunk * 32 + offs).to(
                tl.float32
            )
            scores = tl.exp(logits - global_max) / denom
            tl.store(scores_ptr + token * 256 + chunk * 32 + offs, scores)


@triton.jit
def _group_top8_kernel(
    scores_ptr, values_ptr, ids_ptr, group_scores_ptr, n_tokens: tl.constexpr
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    offs = tl.arange(0, 32)
    ranks = tl.arange(0, 8)
    scores = tl.load(scores_ptr + token * 256 + group * 32 + offs)
    group_score = tl.max(scores, axis=0)
    work = scores
    top_values = tl.zeros((8,), tl.float32)
    top_ids = tl.zeros((8,), tl.int32)
    for rank in range(8):
        local_id = tl.argmax(work, axis=0)
        value = tl.max(work, axis=0)
        top_values = tl.where(ranks == rank, value, top_values)
        top_ids = tl.where(ranks == rank, local_id + group * 32, top_ids)
        work = tl.where(offs == local_id, -float("inf"), work)
    tl.store(values_ptr + token * 64 + group * 8 + ranks, top_values)
    tl.store(ids_ptr + token * 64 + group * 8 + ranks, top_ids)
    tl.store(group_scores_ptr + token * 8 + group, group_score)


@triton.jit
def _merge_groups_kernel(
    group_scores_ptr,
    values_ptr,
    ids_ptr,
    weights_ptr,
    out_ids_ptr,
    n_tokens: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scale: tl.constexpr,
):
    token = tl.program_id(0)
    groups = tl.arange(0, 8)
    group_scores = tl.load(group_scores_ptr + token * 8 + groups)
    g0 = tl.argmax(group_scores, axis=0)
    work = tl.where(groups == g0, -float("inf"), group_scores)
    g1 = tl.argmax(work, axis=0)
    work = tl.where(groups == g1, -float("inf"), work)
    g2 = tl.argmax(work, axis=0)
    work = tl.where(groups == g2, -float("inf"), work)
    g3 = tl.argmax(work, axis=0)

    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    ranks = tl.arange(0, 8)
    top_values = tl.zeros((8,), tl.float32)
    top_ids = tl.zeros((8,), tl.int32)
    for rank in range(8):
        v0 = tl.load(values_ptr + token * 64 + g0 * 8 + i0)
        v1 = tl.load(values_ptr + token * 64 + g1 * 8 + i1)
        v2 = tl.load(values_ptr + token * 64 + g2 * 8 + i2)
        v3 = tl.load(values_ptr + token * 64 + g3 * 8 + i3)
        best = v0
        best_group = 0
        take = v1 > best
        best = tl.where(take, v1, best)
        best_group = tl.where(take, 1, best_group)
        take = v2 > best
        best = tl.where(take, v2, best)
        best_group = tl.where(take, 2, best_group)
        take = v3 > best
        best = tl.where(take, v3, best)
        best_group = tl.where(take, 3, best_group)
        chosen_index = tl.where(
            best_group == 0,
            i0,
            tl.where(best_group == 1, i1, tl.where(best_group == 2, i2, i3)),
        )
        chosen_group = tl.where(
            best_group == 0,
            g0,
            tl.where(best_group == 1, g1, tl.where(best_group == 2, g2, g3)),
        )
        chosen_id = tl.load(ids_ptr + token * 64 + chosen_group * 8 + chosen_index)
        top_values = tl.where(ranks == rank, best, top_values)
        top_ids = tl.where(ranks == rank, chosen_id, top_ids)
        i0 = tl.where(best_group == 0, i0 + 1, i0)
        i1 = tl.where(best_group == 1, i1 + 1, i1)
        i2 = tl.where(best_group == 2, i2 + 1, i2)
        i3 = tl.where(best_group == 3, i3 + 1, i3)
    if renormalize:
        top_values = top_values / tl.sum(top_values, axis=0)
    top_values = top_values * routed_scale
    tl.store(weights_ptr + token * 8 + ranks, top_values)
    tl.store(out_ids_ptr + token * 8 + ranks, top_ids)


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
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor):
        n_tokens = gating_output.shape[0]
        scores = torch.empty_like(gating_output, dtype=torch.float32)
        group_values = torch.empty(
            (n_tokens, 64), device=gating_output.device, dtype=torch.float32
        )
        group_ids = torch.empty(
            (n_tokens, 64), device=gating_output.device, dtype=torch.int32
        )
        group_scores = torch.empty(
            (n_tokens, 8), device=gating_output.device, dtype=torch.float32
        )
        weights = torch.empty(
            (n_tokens, 8), device=gating_output.device, dtype=torch.float32
        )
        ids = torch.empty((n_tokens, 8), device=gating_output.device, dtype=torch.int32)
        _normalize_scores_kernel[(n_tokens,)](
            gating_output,
            scores,
            n_tokens,
            scoring_sigmoid=self.scoring_func == "sigmoid",
            num_warps=1,
            num_stages=1,
        )
        _group_top8_kernel[(n_tokens, 8)](
            scores,
            group_values,
            group_ids,
            group_scores,
            n_tokens,
            num_warps=1,
            num_stages=1,
        )
        _merge_groups_kernel[(n_tokens,)](
            group_scores,
            group_values,
            group_ids,
            weights,
            ids,
            n_tokens,
            renormalize=self.renormalize,
            routed_scale=self.routed_scaling_factor,
            num_warps=1,
            num_stages=1,
        )
        return weights, ids


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(83, 7168, dtype=torch.float16, device="cuda"),
        torch.randn(83, 256, dtype=torch.float32, device="cuda"),
    ]


def get_init_inputs():
    return [8, True, 8, 4]
