import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize

        # w1: gate+up fused projection  [E, 2*intermediate, hidden]
        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        # w2: down projection  [E, hidden, intermediate]
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H]
        router_logits: torch.Tensor,   # [T, E]  float32
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # --- routing ---
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)  # [T, top_k]
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)  # [T, top_k]

        # --- expert dispatch (vectorised per-expert scatter) ---
        # Flatten: treat each (token, k) pair as an independent row
        flat_ids = topk_ids.view(-1)                          # [T*top_k]
        flat_w  = topk_weights.view(-1)                       # [T*top_k]
        x_rep   = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.top_k, -1)
            .reshape(-1, self.hidden_size)
        )  # [T*top_k, H]

        w1 = self.w1.to(dtype)   # [E, 2*I, H]
        w2 = self.w2.to(dtype)   # [E, H, I]

        expert_out = torch.zeros_like(x_rep)
        for e in range(self.num_experts):
            mask = flat_ids == e
            if not mask.any():
                continue
            x_e    = x_rep[mask]                        # [n_e, H]
            gate_up = x_e @ w1[e].T                    # [n_e, 2*I]
            gate, up = gate_up.chunk(2, dim=-1)
            act    = F.silu(gate) * up                  # [n_e, I]
            expert_out[mask] = act @ w2[e].T            # [n_e, H]

        # --- weighted reduction ---
        expert_out = expert_out * flat_w.unsqueeze(-1)
        return expert_out.view(num_tokens, self.top_k, self.hidden_size).sum(dim=1)


def get_inputs():
    # hidden_states: [num_tokens, hidden_size], float16
    # router_logits:  [num_tokens, num_experts], float32
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cpu")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cpu")
    return [hidden_states, router_logits]


def get_init_inputs():
    # num_experts, top_k, hidden_size, intermediate_size
    return [8, 2, 128, 64]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "shape"):
                print(o.shape)
    else:
        print(out.shape)
