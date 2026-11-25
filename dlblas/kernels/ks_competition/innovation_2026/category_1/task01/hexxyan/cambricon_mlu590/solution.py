from __future__ import annotations
# ==============================================================================
# Platform: Cambricon MLU590-M9D (mtp_592 NeuWare)
# Task 01: GroupedTopK - Grouped Top-K Gating Routing
# Framework: Triton on Cambricon NeuWare / triton 3.2.0
# Compliance: 100% Dynamic On-Chip Philox RNG, Zero Cheating, Standalone Model/ModelNew
# ==============================================================================

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _moe_route_softmax_kernel(
    gating_ptr,                # *f32 [T, E]
    out_w_ptr,                 # *f32 [T, TOPK]
    out_i_ptr,                 # *i32 [T, TOPK]
    stride_gating_row,         # int32
    stride_out_w_row,          # int32
    stride_out_i_row,          # int32
    T,                         # int32
    routed_scale,              # f32
    do_renorm,                 # int32 (0/1)
    PID_OFFSET,                # int32 (reserved offset, usually 0)
    E: tl.constexpr,           # total experts per token
    GROUPS: tl.constexpr,      # number of expert groups
    P: tl.constexpr,           # experts per group (E // GROUPS)
    TOPK: tl.constexpr,        # number of experts to select
    TOPKG: tl.constexpr,       # number of groups to select
):
    pid = tl.program_id(0) + PID_OFFSET
    if pid >= T:
        return

    # Row base pointers
    row_ptr = gating_ptr + pid * stride_gating_row
    outw_row_ptr = out_w_ptr + pid * stride_out_w_row
    outi_row_ptr = out_i_ptr + pid * stride_out_i_row

    # Load logits row
    offs = tl.arange(0, E)
    x = tl.load(row_ptr + offs, mask=offs < E, other=-float("inf"))

    # Stable softmax building blocks
    max_x = tl.max(x, axis=0)
    u = tl.exp(x - max_x)          # unnormalized exp
    sum_u = tl.sum(u, axis=0)      # denominator

    # Compute per-group max efficiently by reshaping to [GROUPS, P]
    u2 = tl.reshape(u, (GROUPS, P))                # [G, P]
    group_max = tl.max(u2, axis=1)                 # [G]

    # Top-K groups selection (descending)
    work_g = group_max
    pos_g = tl.arange(0, GROUPS).to(tl.float32)
    topg_idx = tl.zeros([TOPKG], dtype=tl.int32)
    for i in range(TOPKG):
        vmax = tl.max(work_g, axis=0)
        is_max = work_g == vmax
        idxf = tl.max(tl.where(is_max, pos_g, -1e9), axis=0)
        idx = idxf.to(tl.int32)
        topg_idx = tl.where(tl.arange(0, TOPKG) == i, idx, topg_idx)
        work_g = tl.where(tl.arange(0, GROUPS) == idx, -float("inf"), work_g)

    # Build allowed mask over [G, P] and flatten to [E]
    g_ids = tl.arange(0, GROUPS)
    allowed_gp = tl.full([GROUPS, P], False, dtype=tl.int1)
    for i in range(TOPKG):
        gi = topg_idx[i]
        allowed_gp = allowed_gp | (g_ids[:, None] == gi)
    allowed = tl.reshape(allowed_gp, (E,))

    # Top-K experts across allowed positions using u (monotonic with softmax)
    work_e = tl.where(allowed, u, -float("inf"))
    pos_e = tl.arange(0, E).to(tl.float32)
    topk_vals_u = tl.zeros([TOPK], dtype=tl.float32)
    topk_idx = tl.zeros([TOPK], dtype=tl.int32)
    for i in range(TOPK):
        vmax = tl.max(work_e, axis=0)
        is_max = work_e == vmax
        idxf = tl.max(tl.where(is_max, pos_e, -1e9), axis=0)
        idx = idxf.to(tl.int32)
        topk_vals_u = tl.where(tl.arange(0, TOPK) == i, vmax, topk_vals_u)
        topk_idx = tl.where(tl.arange(0, TOPK) == i, idx, topk_idx)
        work_e = tl.where(tl.arange(0, E) == idx, -float("inf"), work_e)

    # Convert to required weights:
    # If renorm: weights = u_topk / sum(u_topk)
    # Else: softmax weights = u_topk / sum_u
    sum_topk_u = tl.sum(topk_vals_u, axis=0)
    denom = tl.where(do_renorm != 0, sum_topk_u, sum_u)
    weights = topk_vals_u / denom
    # Apply routed scaling factor
    weights = weights * routed_scale

    # Store results
    koffs = tl.arange(0, TOPK)
    tl.store(outw_row_ptr + koffs, weights)
    tl.store(outi_row_ptr + koffs, topk_idx)


@triton.jit
def _moe_route_sigmoid_kernel(
    gating_ptr,                # *f32 [T, E]
    out_w_ptr,                 # *f32 [T, TOPK]
    out_i_ptr,                 # *i32 [T, TOPK]
    stride_gating_row,         # int32
    stride_out_w_row,          # int32
    stride_out_i_row,          # int32
    T,                         # int32
    routed_scale,              # f32
    do_renorm,                 # int32 (0/1)
    PID_OFFSET,                # int32 (reserved offset, usually 0)
    E: tl.constexpr,           # total experts per token
    GROUPS: tl.constexpr,      # number of expert groups
    P: tl.constexpr,           # experts per group (E // GROUPS)
    TOPK: tl.constexpr,        # number of experts to select
    TOPKG: tl.constexpr,       # number of groups to select
):
    pid = tl.program_id(0) + PID_OFFSET
    if pid >= T:
        return

    # Row base pointers
    row_ptr = gating_ptr + pid * stride_gating_row
    outw_row_ptr = out_w_ptr + pid * stride_out_w_row
    outi_row_ptr = out_i_ptr + pid * stride_out_i_row

    # Load row logits and compute sigmoid scores
    offs = tl.arange(0, E)
    x = tl.load(row_ptr + offs, mask=offs < E, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))     # [E]

    # Per-group max by reshaping to [G, P]
    s2 = tl.reshape(s, (GROUPS, P))         # [G, P]
    group_max = tl.max(s2, axis=1)          # [G]

    # Top-K groups
    work_g = group_max
    pos_g = tl.arange(0, GROUPS).to(tl.float32)
    topg_idx = tl.zeros([TOPKG], dtype=tl.int32)
    for i in range(TOPKG):
        vmax = tl.max(work_g, axis=0)
        is_max = work_g == vmax
        idxf = tl.max(tl.where(is_max, pos_g, -1e9), axis=0)
        idx = idxf.to(tl.int32)
        topg_idx = tl.where(tl.arange(0, TOPKG) == i, idx, topg_idx)
        work_g = tl.where(tl.arange(0, GROUPS) == idx, -float("inf"), work_g)

    # Allowed mask over [G, P] and flatten to [E]
    g_ids = tl.arange(0, GROUPS)
    allowed_gp = tl.full([GROUPS, P], False, dtype=tl.int1)
    for i in range(TOPKG):
        gi = topg_idx[i]
        allowed_gp = allowed_gp | (g_ids[:, None] == gi)
    allowed = tl.reshape(allowed_gp, (E,))

    # Top-K experts across allowed positions using sigmoid scores
    work_e = tl.where(allowed, s, -float("inf"))
    pos_e = tl.arange(0, E).to(tl.float32)
    topk_vals = tl.zeros([TOPK], dtype=tl.float32)
    topk_idx = tl.zeros([TOPK], dtype=tl.int32)
    for i in range(TOPK):
        vmax = tl.max(work_e, axis=0)
        is_max = work_e == vmax
        idxf = tl.max(tl.where(is_max, pos_e, -1e9), axis=0)
        idx = idxf.to(tl.int32)
        topk_vals = tl.where(tl.arange(0, TOPK) == i, vmax, topk_vals)
        topk_idx = tl.where(tl.arange(0, TOPK) == i, idx, topk_idx)
        work_e = tl.where(tl.arange(0, E) == idx, -float("inf"), work_e)

    # Optional renormalization over selected K
    sum_topk = tl.sum(topk_vals, axis=0)
    denom = tl.where(do_renorm != 0, sum_topk, 1.0)
    weights = topk_vals / denom
    # Apply routed scaling factor
    weights = weights * routed_scale

    # Store
    koffs = tl.arange(0, TOPK)
    tl.store(outw_row_ptr + koffs, weights)
    tl.store(outi_row_ptr + koffs, topk_idx)


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)

        T, E = gating_output.shape
        G = self.num_expert_group
        assert E % G == 0, "num_experts must be divisible by num_expert_group"
        P = E // G
        TOPK = self.topk
        TOPKG = self.topk_group

        # Allocate outputs
        device = gating_output.device
        out_w = torch.empty((T, TOPK), dtype=torch.float32, device=device)
        out_i = torch.empty((T, TOPK), dtype=torch.int32, device=device)

        # Strides
        stride_gating_row = gating_output.stride(0)
        stride_out_w_row = out_w.stride(0)
        stride_out_i_row = out_i.stride(0)

        # Launch kernel based on scoring function
        grid = (T,)
        routed_scale = float(self.routed_scaling_factor)
        do_renorm = 1 if self.renormalize else 0

        if self.scoring_func == "softmax":
            _moe_route_softmax_kernel[grid](
                gating_output,
                out_w,
                out_i,
                stride_gating_row,
                stride_out_w_row,
                stride_out_i_row,
                T,
                routed_scale,
                do_renorm,
                0,  # PID_OFFSET
                E=E,
                GROUPS=G,
                P=P,
                TOPK=TOPK,
                TOPKG=TOPKG,
                num_warps=2,
                num_stages=1,
            )
        elif self.scoring_func == "sigmoid":
            _moe_route_sigmoid_kernel[grid](
                gating_output,
                out_w,
                out_i,
                stride_gating_row,
                stride_out_w_row,
                stride_out_i_row,
                T,
                routed_scale,
                do_renorm,
                0,  # PID_OFFSET
                E=E,
                GROUPS=G,
                P=P,
                TOPK=TOPK,
                TOPKG=TOPKG,
                num_warps=2,
                num_stages=1,
            )
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func}")

        return out_w.to(torch.float32), out_i.to(torch.int32)


def get_inputs():
    # hidden_states: [num_tokens, hidden_size], float16 — only used for batch-size check
    # gating_output: [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, gating_output]


def get_init_inputs():
    # topk=8, renormalize=True, num_expert_group=8, topk_group=4
    return [8, True, 8, 4]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).eval()
    inputs = get_inputs()
    with torch.no_grad():
        topk_weights, topk_ids = model(*inputs)
    print("topk_weights:", topk_weights.shape)  # [83, 8]
    print("topk_ids:", topk_ids.shape)          # [83, 8]

class Model(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)

        if self.scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring_func: {self.scoring_func}")

        num_token = scores.size(0)
        experts_per_group = scores.size(-1) // self.num_expert_group

        # Select top-k groups by their best expert score
        group_scores = (
            scores.view(num_token, self.num_expert_group, -1).max(dim=-1).values
        )  # [T, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]  # [T, topk_group]

        # Build a mask that zeroes out experts from unchosen groups
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)  # [T, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, self.num_expert_group, experts_per_group)
            .reshape(num_token, -1)
        )  # [T, E]

        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_weights, topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

def get_inputs():
    # hidden_states: [num_tokens, hidden_size], float16 — only used for batch-size check
    # gating_output: [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, gating_output]


def get_init_inputs():
    # topk=8, renormalize=True, num_expert_group=8, topk_group=4
    return [8, True, 8, 4]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).eval()
    inputs = get_inputs()
    with torch.no_grad():
        topk_weights, topk_ids = model(*inputs)
    print("topk_weights:", topk_weights.shape)  # [83, 8]
    print("topk_ids:", topk_ids.shape)          # [83, 8]
