import torch
import torch.nn as nn


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
    print(topk_weights.shape)   # [83, 8]
    print(topk_ids.shape)       # [83, 8]
