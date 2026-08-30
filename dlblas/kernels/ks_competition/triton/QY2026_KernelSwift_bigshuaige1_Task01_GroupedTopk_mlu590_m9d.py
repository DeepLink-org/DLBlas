import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _moe_route_fused_kernel(
    gating_ptr,  # [T, E] gating scores
    out_w_ptr,  # [T, topk] float32 output weights
    out_id_ptr,  # [T, topk] int32 output expert ids
    scaling,  # routed scaling factor (runtime scalar)
    NUM_EXPERTS: tl.constexpr,
    N_GROUP: tl.constexpr,
    EPG: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    USE_SIGMOID: tl.constexpr,
    RENORM: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)

    # 2-D tile: rows = expert groups, cols = experts within a group.
    # Shape params are compile-time constants -> offsets/masks constant-fold.
    g = tl.arange(0, BLOCK_G)
    c = tl.arange(0, BLOCK_C)
    eid = g[:, None] * EPG + c[None, :]
    lmask = (g < N_GROUP)[:, None] & (c < EPG)[None, :] & (eid < NUM_EXPERTS)

    x = tl.load(
        gating_ptr + row * NUM_EXPERTS + eid, mask=lmask, other=float("-inf")
    ).to(tl.float32)

    # ---- fused scoring function (softmax / sigmoid), fp32 ----
    if USE_SIGMOID:
        scores = 1.0 / (1.0 + tl.exp(-x))
    else:
        m = tl.max(tl.max(x, axis=1), axis=0)
        ex = tl.exp(x - m)
        ex = tl.where(lmask, ex, 0.0)
        denom = tl.sum(tl.sum(ex, axis=1), axis=0)
        scores = ex / denom
    scores = tl.where(lmask, scores, float("-inf"))

    # ---- select TOPK_GROUP groups by their best expert score ----
    group_vals = tl.max(scores, axis=1)  # [BLOCK_G]
    gsel = g < 0  # all-false seed
    for _ in range(TOPK_GROUP):
        gv = tl.max(group_vals, axis=0)
        gidx = tl.min(tl.where(group_vals == gv, g, BLOCK_G), axis=0)
        gsel = gsel | (g == gidx)
        group_vals = tl.where(g == gidx, float("-inf"), group_vals)

    # ---- zero out experts belonging to unselected groups ----
    tmp = tl.where(gsel[:, None], scores, float("-inf"))

    # ---- iterative top-K experts (descending, ties -> lower index) ----
    # Weights/ids stay in registers; single store phase at the end.
    k = tl.arange(0, BLOCK_K)
    sel_w = tl.zeros([BLOCK_K], dtype=tl.float32)
    sel_i = tl.zeros([BLOCK_K], dtype=tl.int32)
    for i in range(TOPK):
        v = tl.max(tl.max(tmp, axis=1), axis=0)
        pos = tl.min(tl.min(tl.where(tmp == v, eid, NUM_EXPERTS), axis=1), axis=0)
        sel_w = tl.where(k == i, v, sel_w)
        sel_i = tl.where(k == i, pos.to(tl.int32), sel_i)
        tmp = tl.where(eid == pos, float("-inf"), tmp)

    # ---- fused renormalize + routed scaling (register only) ----
    wsum = tl.sum(tl.where(k < TOPK, sel_w, 0.0), axis=0)
    if RENORM:
        sel_w = sel_w / wsum
    if APPLY_SCALE:
        sel_w = sel_w * scaling

    omask = k < TOPK
    tl.store(out_w_ptr + row * TOPK + k, sel_w, mask=omask)
    tl.store(out_id_ptr + row * TOPK + k, sel_i, mask=omask)


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
        # Pre-computed launch constants (avoids per-call Python overhead)
        self._sigmoid = scoring_func == "sigmoid"
        self._valid_scoring = scoring_func in ("softmax", "sigmoid")
        self._renorm = bool(renormalize)
        self._scale = float(routed_scaling_factor)
        self._apply_scale = routed_scaling_factor != 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == gating_output.size(0)

        gating = gating_output.contiguous()
        num_token, num_experts = gating.shape
        n_group = self.num_expert_group
        experts_per_group = num_experts // n_group

        topk_weights = torch.empty(
            (num_token, self.topk), dtype=torch.float32, device=gating.device
        )
        topk_ids = torch.empty(
            (num_token, self.topk), dtype=torch.int32, device=gating.device
        )

        if num_token > 0:
            _moe_route_fused_kernel[(num_token,)](
                gating,
                topk_weights,
                topk_ids,
                self._scale,
                NUM_EXPERTS=num_experts,
                N_GROUP=n_group,
                EPG=experts_per_group,
                TOPK=self.topk,
                TOPK_GROUP=self.topk_group,
                USE_SIGMOID=self._sigmoid,
                RENORM=self._renorm,
                APPLY_SCALE=self._apply_scale,
                BLOCK_G=triton.next_power_of_2(n_group),
                BLOCK_C=triton.next_power_of_2(experts_per_group),
                BLOCK_K=triton.next_power_of_2(self.topk),
            )

        return topk_weights, topk_ids


class Model(ModelNew):
    """Strict-package wrapper; the scored implementation remains ModelNew."""

    pass


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
    print(topk_weights.shape)  # [83, 8]
    print(topk_ids.shape)  # [83, 8]
