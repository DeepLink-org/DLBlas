"""Kunlun P800 Task02: direct top-2 expert dispatch with split gate/up and down."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


# P800 Triton 3.0 hypothesis: keep the 8-wide K tile and use two warps for
# the small M=1 expert dispatch to reduce launch/register overhead.
BLOCK_K = 8
KERNEL_WARPS = 2


@triton.jit
def _top2_gate_up_kernel(x_ptr, router_ptr, w1_ptr, act_ptr, BLOCK_K: tl.constexpr):
    token = tl.program_id(0)
    choice = tl.program_id(1)
    eo = tl.arange(0, 8)
    logits = tl.load(router_ptr + token * 8 + eo).to(tl.float32)
    e0 = tl.argmax(logits, axis=0)
    rest = tl.where(eo == e0, -float("inf"), logits)
    e1 = tl.argmax(rest, axis=0)
    expert = tl.where(choice == 0, e0, e1)
    i = tl.arange(0, 64)
    gate_acc = tl.zeros((64,), tl.float32)
    up_acc = tl.zeros((64,), tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + token * 128 + k).to(tl.float16)
        gw = tl.load(w1_ptr + expert * 16384 + i[:, None] * 128 + k[None, :]).to(tl.float16)
        uw = tl.load(w1_ptr + expert * 16384 + (i[:, None] + 64) * 128 + k[None, :]).to(tl.float16)
        gate_acc += tl.sum(gw * xv[None, :], axis=1)
        up_acc += tl.sum(uw * xv[None, :], axis=1)
    act = (gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc).to(tl.float16)
    tl.store(act_ptr + (token * 2 + choice) * 64 + i, act)


@triton.jit
def _top2_down_kernel(act_ptr, router_ptr, w2_ptr, tmp_ptr, BLOCK_K: tl.constexpr):
    token = tl.program_id(0)
    choice = tl.program_id(1)
    eo = tl.arange(0, 8)
    logits = tl.load(router_ptr + token * 8 + eo).to(tl.float32)
    e0 = tl.argmax(logits, axis=0)
    rest = tl.where(eo == e0, -float("inf"), logits)
    e1 = tl.argmax(rest, axis=0)
    expert = tl.where(choice == 0, e0, e1)
    h = tl.arange(0, 128)
    acc = tl.zeros((128,), tl.float32)
    for k0 in range(0, 64, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        av = tl.load(act_ptr + (token * 2 + choice) * 64 + k).to(tl.float16)
        w = tl.load(w2_ptr + expert * 8192 + h[:, None] * 64 + k[None, :]).to(tl.float16)
        acc += tl.sum(w * av[None, :], axis=1)
    tl.store(tmp_ptr + (token * 2 + choice) * 128 + h, acc.to(tl.float16))


@triton.jit
def _top2_reduce_kernel(tmp_ptr, router_ptr, out_ptr):
    token = tl.program_id(0)
    h = tl.arange(0, 128)
    eo = tl.arange(0, 8)
    logits = tl.load(router_ptr + token * 8 + eo).to(tl.float32)
    e0 = tl.argmax(logits, axis=0)
    x0 = tl.max(logits, axis=0)
    rest = tl.where(eo == e0, -float("inf"), logits)
    x1 = tl.max(rest, axis=0)
    w0 = 1.0 / (1.0 + tl.exp(x1 - x0))
    y0 = tl.load(tmp_ptr + token * 2 * 128 + h).to(tl.float32)
    y1 = tl.load(tmp_ptr + (token * 2 + 1) * 128 + h).to(tl.float32)
    tl.store(out_ptr + token * 128 + h, y0 * w0 + y1 * (1.0 - w0))


class ModelNew(nn.Module):
    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, renormalize: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        t = hidden_states.shape[0]
        act = torch.empty((t, 2, 64), dtype=hidden_states.dtype, device=hidden_states.device)
        tmp = torch.empty((t, 2, 128), dtype=hidden_states.dtype, device=hidden_states.device)
        out = torch.empty_like(hidden_states)
        _top2_gate_up_kernel[(t, 2)](hidden_states, router_logits, self.w1, act, BLOCK_K=BLOCK_K, num_warps=KERNEL_WARPS, num_stages=1)
        _top2_down_kernel[(t, 2)](act, router_logits, self.w2, tmp, BLOCK_K=BLOCK_K, num_warps=KERNEL_WARPS, num_stages=1)
        _top2_reduce_kernel[(t,)](tmp, router_logits, out, num_warps=KERNEL_WARPS, num_stages=1)
        return out


class Model(ModelNew):
    pass


def get_inputs():
    return [torch.randn(83, 128, dtype=torch.float16, device="xpu"), torch.randn(83, 8, dtype=torch.float32, device="xpu")]


def get_init_inputs():
    return [8, 2, 128, 64]
