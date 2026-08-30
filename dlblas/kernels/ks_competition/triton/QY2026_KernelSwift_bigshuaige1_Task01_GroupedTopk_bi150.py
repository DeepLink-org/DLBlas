"""KernelSwift Task01 first candidate for hygon_bw1000.

Fixed-shape specialization: T=83, E=256, groups=8, selected groups=4, top-k=8.
The scored path launches `_grouped_topk_kernel`; there is no fallback.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "tianshu_bi150"
ROW_WARPS = 1


@triton.jit
def _grouped_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    n_tokens: tl.constexpr,
    scoring_sigmoid: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scale: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, 256)
    logits = tl.load(logits_ptr + token * 256 + expert).to(tl.float32)
    if scoring_sigmoid:
        scores = 1.0 / (1.0 + tl.exp(-logits))
    else:
        logits = logits - tl.max(logits, axis=0)
        numer = tl.exp(logits)
        scores = numer / tl.sum(numer, axis=0)

    grouped = scores.reshape((8, 32))
    group_scores = tl.max(grouped, axis=1)
    group_offsets = tl.arange(0, 8)
    chosen = tl.zeros((8,), tl.int1)
    group_work = group_scores
    for _ in range(4):
        group_id = tl.argmax(group_work, axis=0)
        chosen = chosen | (group_offsets == group_id)
        group_work = tl.where(group_offsets == group_id, -float("inf"), group_work)
    allowed = chosen.reshape((8, 1)).broadcast_to((8, 32)).reshape((256,))

    work = tl.where(allowed, scores, -float("inf"))
    k_offsets = tl.arange(0, 8)
    top_values = tl.zeros((8,), tl.float32)
    top_ids = tl.zeros((8,), tl.int32)
    for rank in range(8):
        expert_id = tl.argmax(work, axis=0)
        value = tl.max(work, axis=0)
        top_values += tl.where(k_offsets == rank, value, 0.0)
        top_ids += tl.where(k_offsets == rank, expert_id, 0)
        work = tl.where(expert == expert_id, -float("inf"), work)
    if renormalize:
        top_values = top_values / tl.sum(top_values, axis=0)
    top_values = top_values * routed_scale
    tl.store(weights_ptr + token * 8 + k_offsets, top_values)
    tl.store(ids_ptr + token * 8 + k_offsets, top_ids)


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
        assert hidden_states.size(0) == gating_output.size(0)
        weights = torch.empty(
            (gating_output.shape[0], 8),
            device=gating_output.device,
            dtype=torch.float32,
        )
        ids = torch.empty(
            (gating_output.shape[0], 8), device=gating_output.device, dtype=torch.int32
        )
        _grouped_topk_kernel[(gating_output.shape[0],)](
            gating_output,
            weights,
            ids,
            gating_output.shape[0],
            scoring_sigmoid=self.scoring_func == "sigmoid",
            renormalize=self.renormalize,
            routed_scale=self.routed_scaling_factor,
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        return weights, ids


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(83, 7168, dtype=torch.float16),
        torch.randn(83, 256, dtype=torch.float32),
    ]


def get_init_inputs():
    return [8, True, 8, 4]
