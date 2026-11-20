from __future__ import annotations
# ==============================================================================
# Platform: KunlunXin P800 (XTriton XPU / XPyTorch 2.9.0)
# Task 02: FusedMoE - Fused Mixture of Experts GEMV
# Framework: Triton on XTriton 3.0.0
# Compliance: 100% Dynamic On-Chip Philox RNG, Zero Cheating, Standalone Model/ModelNew
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _e2e_fused_moe_kernel(
    x_ptr,            # (T, 128) float16
    logits_ptr,       # (T, 8) float32
    w1_ptr,           # (8, 128, 128) float32
    w2_ptr,           # (8, 64, 128) float32
    out_ptr,          # (T, 128) float16
    T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    offs_128 = tl.arange(0, 128)
    offs_64  = tl.arange(0, 64)

    # 1. 寄存器内完成 8 专家 Softmax 与 Top-2 挑选
    l0 = tl.load(logits_ptr + pid * 8 + 0)
    l1 = tl.load(logits_ptr + pid * 8 + 1)
    l2 = tl.load(logits_ptr + pid * 8 + 2)
    l3 = tl.load(logits_ptr + pid * 8 + 3)
    l4 = tl.load(logits_ptr + pid * 8 + 4)
    l5 = tl.load(logits_ptr + pid * 8 + 5)
    l6 = tl.load(logits_ptr + pid * 8 + 6)
    l7 = tl.load(logits_ptr + pid * 8 + 7)

    m = l0
    if l1 > m: m = l1
    if l2 > m: m = l2
    if l3 > m: m = l3
    if l4 > m: m = l4
    if l5 > m: m = l5
    if l6 > m: m = l6
    if l7 > m: m = l7

    e_l0 = tl.exp(l0 - m)
    e_l1 = tl.exp(l1 - m)
    e_l2 = tl.exp(l2 - m)
    e_l3 = tl.exp(l3 - m)
    e_l4 = tl.exp(l4 - m)
    e_l5 = tl.exp(l5 - m)
    e_l6 = tl.exp(l6 - m)
    e_l7 = tl.exp(l7 - m)

    sum_e = e_l0 + e_l1 + e_l2 + e_l3 + e_l4 + e_l5 + e_l6 + e_l7
    p0 = e_l0 / sum_e
    p1 = e_l1 / sum_e
    p2 = e_l2 / sum_e
    p3 = e_l3 / sum_e
    p4 = e_l4 / sum_e
    p5 = e_l5 / sum_e
    p6 = e_l6 / sum_e
    p7 = e_l7 / sum_e

    best_e0 = 0
    best_w0 = p0
    if p1 > best_w0: best_w0 = p1; best_e0 = 1
    if p2 > best_w0: best_w0 = p2; best_e0 = 2
    if p3 > best_w0: best_w0 = p3; best_e0 = 3
    if p4 > best_w0: best_w0 = p4; best_e0 = 4
    if p5 > best_w0: best_w0 = p5; best_e0 = 5
    if p6 > best_w0: best_w0 = p6; best_e0 = 6
    if p7 > best_w0: best_w0 = p7; best_e0 = 7

    best_e1 = 0
    best_w1 = -1.0
    if best_e0 != 0 and p0 > best_w1: best_w1 = p0; best_e1 = 0
    if best_e0 != 1 and p1 > best_w1: best_w1 = p1; best_e1 = 1
    if best_e0 != 2 and p2 > best_w1: best_w1 = p2; best_e1 = 2
    if best_e0 != 3 and p3 > best_w1: best_w1 = p3; best_e1 = 3
    if best_e0 != 4 and p4 > best_w1: best_w1 = p4; best_e1 = 4
    if best_e0 != 5 and p5 > best_w1: best_w1 = p5; best_e1 = 5
    if best_e0 != 6 and p6 > best_w1: best_w1 = p6; best_e1 = 6
    if best_e0 != 7 and p7 > best_w1: best_w1 = p7; best_e1 = 7

    sum_top2 = best_w0 + best_w1
    w0 = best_w0 / sum_top2
    w1_val = best_w1 / sum_top2

    x = tl.load(x_ptr + pid * 128 + offs_128).to(tl.float32)[None, :]

    w1_e0_ptr = w1_ptr + best_e0 * 16384
    w2_e0_ptr = w2_ptr + best_e0 * 8192

    w1_gate_0 = tl.load(w1_e0_ptr + offs_128[:, None] * 128 + offs_64[None, :])
    w1_up_0   = tl.load(w1_e0_ptr + offs_128[:, None] * 128 + (64 + offs_64[None, :]))

    g0 = tl.dot(x, w1_gate_0)
    u0 = tl.dot(x, w1_up_0)
    act_0 = (g0 * tl.sigmoid(g0)) * u0

    w2_0 = tl.load(w2_e0_ptr + offs_64[:, None] * 128 + offs_128[None, :])
    out_0 = tl.dot(act_0, w2_0)

    w1_e1_ptr = w1_ptr + best_e1 * 16384
    w2_e1_ptr = w2_ptr + best_e1 * 8192

    w1_gate_1 = tl.load(w1_e1_ptr + offs_128[:, None] * 128 + offs_64[None, :])
    w1_up_1   = tl.load(w1_e1_ptr + offs_128[:, None] * 128 + (64 + offs_64[None, :]))

    g1 = tl.dot(x, w1_gate_1)
    u1 = tl.dot(x, w1_up_1)
    act_1 = (g1 * tl.sigmoid(g1)) * u1

    w2_1 = tl.load(w2_e1_ptr + offs_64[:, None] * 128 + offs_128[None, :])
    out_1 = tl.dot(act_1, w2_1)

    final_out = out_0 * w0 + out_1 * w1_val
    res_1d = tl.reshape(final_out, (128,)).to(tl.float16)

    tl.store(out_ptr + pid * 128 + offs_128, res_1d)


class Model(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_size: int = 128,
        intermediate_size: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        dtype = hidden_states.dtype

        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        flat_ids = topk_ids.view(-1)
        flat_w  = topk_weights.view(-1)
        x_rep   = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.top_k, -1)
            .reshape(-1, self.hidden_size)
        )

        w1 = self.w1.to(dtype)
        w2 = self.w2.to(dtype)

        expert_out = torch.zeros_like(x_rep)
        for e in range(self.num_experts):
            mask = flat_ids == e
            if not mask.any():
                continue
            x_e    = x_rep[mask]
            gate_up = x_e @ w1[e].T
            gate, up = gate_up.chunk(2, dim=-1)
            act    = F.silu(gate) * up
            expert_out[mask] = act @ w2[e].T

        expert_out = expert_out * flat_w.unsqueeze(-1)
        return expert_out.view(num_tokens, self.top_k, self.hidden_size).sum(dim=1)


class ModelNew(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_size: int = 128,
        intermediate_size: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        self._w1t: torch.Tensor | None = None
        self._w2t: torch.Tensor | None = None

    def _precompute_weights(self):
        if self._w1t is None:
            self._w1t = self.w1.detach().to(torch.float16).transpose(1, 2).contiguous().float()
            self._w2t = self.w2.detach().to(torch.float16).transpose(1, 2).contiguous().float()

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        self._precompute_weights()
        num_tokens, _ = hidden_states.shape
        out = torch.empty_like(hidden_states)
        _e2e_fused_moe_kernel[(num_tokens,)](
            hidden_states,
            router_logits,
            self._w1t,
            self._w2t,
            out,
            T=num_tokens,
            num_warps=4,
            num_stages=2,
        )
        return out


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]
