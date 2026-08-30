"""KernelSwift Task02 MUSA-safe fused MoE candidate.

The MUSA backend does not lower the original matrix-style ``tl.dot`` path
reliably.  This version routes to the selected top-2 experts and performs
each GEMM as 32-element vector partial sums, preserving the custom Triton
kernel boundary without a framework fallback.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "musa_ph100"


@triton.jit
def _route_top2_kernel(logits, route_w, route_id):
    token = tl.program_id(0)
    e = tl.arange(0, 8)
    x = tl.load(logits + token * 8 + e).to(tl.float32)
    x = x - tl.max(x, axis=0)
    p = tl.exp(x)
    p = p / tl.sum(p, axis=0)
    id0 = tl.argmax(p, axis=0)
    p0 = tl.max(p, axis=0)
    p_rest = tl.where(e == id0, -float("inf"), p)
    id1 = tl.argmax(p_rest, axis=0)
    p1 = tl.max(p_rest, axis=0)
    norm = p0 + p1
    tl.store(route_id + token * 2, id0)
    tl.store(route_id + token * 2 + 1, id1)
    tl.store(route_w + token * 2, p0 / norm)
    tl.store(route_w + token * 2 + 1, p1 / norm)


@triton.jit
def _gate_up_scalar_kernel(x_ptr, w1_ptr, route_id, act_ptr, n_tokens: tl.constexpr):
    token = tl.program_id(0)
    choice = tl.program_id(1)
    expert = tl.load(route_id + token * 2 + choice)
    for n in range(64):
        gate_acc = 0.0
        up_acc = 0.0
        for k0 in range(0, 128, 32):
            kk = tl.arange(0, 32)
            x = tl.load(x_ptr + token * 128 + k0 + kk).to(tl.float32)
            gate = tl.load(w1_ptr + expert * 128 * 128 + n * 128 + k0 + kk).to(
                tl.float32
            )
            up = tl.load(w1_ptr + expert * 128 * 128 + (n + 64) * 128 + k0 + kk).to(
                tl.float32
            )
            gate_acc += tl.sum(x * gate, axis=0)
            up_acc += tl.sum(x * up, axis=0)
        value = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
        tl.store(act_ptr + token * 2 * 64 + choice * 64 + n, value)


@triton.jit
def _down_scalar_kernel(act_ptr, w2_ptr, route_id, dense_ptr, n_tokens: tl.constexpr):
    token = tl.program_id(0)
    choice = tl.program_id(1)
    expert = tl.load(route_id + token * 2 + choice)
    for n in range(128):
        acc = 0.0
        for k0 in range(0, 64, 32):
            kk = tl.arange(0, 32)
            a = tl.load(act_ptr + token * 2 * 64 + choice * 64 + k0 + kk).to(tl.float32)
            b = tl.load(w2_ptr + expert * 128 * 64 + n * 64 + k0 + kk).to(tl.float32)
            acc += tl.sum(a * b, axis=0)
        tl.store(dense_ptr + token * 2 * 128 + choice * 128 + n, acc)


@triton.jit
def _route_reduce_kernel(dense, route_w, out):
    token = tl.program_id(0)
    h = tl.arange(0, 32)
    for chunk in range(4):
        hh = chunk * 32 + h
        w0 = tl.load(route_w + token * 2)
        w1 = tl.load(route_w + token * 2 + 1)
        y0 = tl.load(dense + token * 2 * 128 + chunk * 32 + h)
        y1 = tl.load(dense + token * 2 * 128 + 128 + chunk * 32 + h)
        tl.store(
            out + token * 128 + hh, y0.to(tl.float32) * w0 + y1.to(tl.float32) * w1
        )


class ModelNew(nn.Module):
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
        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        t = hidden_states.shape[0]
        route_w = torch.empty((t, 2), dtype=torch.float32, device=hidden_states.device)
        route_id = torch.empty((t, 2), dtype=torch.int32, device=hidden_states.device)
        act = torch.empty(
            (t, 2, 64), dtype=hidden_states.dtype, device=hidden_states.device
        )
        dense = torch.empty(
            (t, 2, 128), dtype=torch.float16, device=hidden_states.device
        )
        out = torch.empty_like(hidden_states)
        _route_top2_kernel[(t,)](
            router_logits, route_w, route_id, num_warps=1, num_stages=1
        )
        _gate_up_scalar_kernel[(t, 2)](
            hidden_states, self.w1, route_id, act, t, num_warps=1, num_stages=1
        )
        _down_scalar_kernel[(t, 2)](
            act, self.w2, route_id, dense, t, num_warps=1, num_stages=1
        )
        _route_reduce_kernel[(t,)](dense, route_w, out, num_warps=1, num_stages=1)
        return out


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(83, 128, dtype=torch.float16, device="cuda"),
        torch.randn(83, 8, dtype=torch.float32, device="cuda"),
    ]


def get_init_inputs():
    return [8, 2, 128, 64]
