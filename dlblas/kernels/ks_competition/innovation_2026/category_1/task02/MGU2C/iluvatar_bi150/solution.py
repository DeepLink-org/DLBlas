"""A one-launch Triton implementation of the small routed SwiGLU MoE."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


_CHIP_CONFIGS = {
    "auto": (8,),
    "ascend_a2": (1,),
    "metax_c500": (8,),
    "iluvatar_bi150": (8,),
    "enflame_s60": (4,),
    "thead_810e": (4,),
    "cambricon_mlu590": (4,),
    "hygon_bw1000": (8,),
    "mthreads_ph100": (8,),
    "kunlun_p800": (4,),
    "biren_br106m": (8,),
}


_VALID_WARPS = (1, 2, 4, 8)
_VALID_IMPLEMENTATIONS = ("fused", "split", "expert")
_VALID_OUT_BLOCKS = (32, 64, 128)
_VALID_TOKEN_BLOCKS = (16, 32, 64)


def _env_choice(name, choices, default):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    for choice in choices:
        if value == str(choice):
            return choice
    return default


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "iluvatar_bi150").strip().lower()
    default_warps = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])[0]
    num_warps = _env_choice(
        "KERNELSWIFT_TASK02_WARPS", _VALID_WARPS, default_warps
    )
    implementation = _env_choice(
        "KERNELSWIFT_TASK02_IMPL", _VALID_IMPLEMENTATIONS, "fused"
    )
    out_block = _env_choice(
        "KERNELSWIFT_TASK02_OUT_BLOCK", _VALID_OUT_BLOCKS, 64
    )
    token_block = _env_choice(
        "KERNELSWIFT_TASK02_TOKEN_BLOCK", _VALID_TOKEN_BLOCKS, 32
    )
    return num_warps, implementation, out_block, token_block


def _tensor_layout_key(tensor):
    """Describe every property that can affect a compiled specialization.

    Accelerator allocations are at least 16-byte aligned.  A view's
    ``storage_offset`` therefore determines its pointer-alignment class without
    putting a transient ``data_ptr`` in the cache key.
    """
    alignment_elements = max(1, 16 // tensor.element_size())
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset() % alignment_elements,
    )


@triton.jit
def _fused_moe_kernel(
    hidden_states,
    router_logits,
    w1,
    w2,
    output,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_r_t: tl.constexpr,
    stride_r_e: tl.constexpr,
    stride_w1_e: tl.constexpr,
    stride_w1_i: tl.constexpr,
    stride_w1_h: tl.constexpr,
    stride_w2_e: tl.constexpr,
    stride_w2_h: tl.constexpr,
    stride_w2_i: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    ROUTE_CACHE: tl.constexpr,
    ROUTE_LOGISTIC: tl.constexpr,
    USE_FAST_LIBDEVICE: tl.constexpr,
):
    token = tl.program_id(0)
    expert_lane = tl.arange(0, BLOCK_E)
    hidden_lane = tl.arange(0, BLOCK_H)
    inter_lane = tl.arange(0, BLOCK_I)

    logits = tl.load(
        router_logits + token * stride_r_t + expert_lane * stride_r_e,
        mask=expert_lane < NUM_EXPERTS,
        other=-float("inf"),
    ).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    # Softmax is strictly increasing, so Top-K is performed on logits.  With
    # renormalization the full-router denominator cancels and only TOP_K
    # exponentials are needed for the final stable softmax.
    if RENORMALIZE:
        if ROUTE_CACHE:
            # The official case is exactly renormalized Top-2.  Select both
            # routes once and retain the ids/logits in registers; the original
            # path below selected the same pair again while computing the two
            # experts.  Keep the identical leftmost tie rule and route order.
            route_expert0 = tl.argmax(
                logits, axis=0, tie_break_left=True
            )
            route_logit0 = tl.max(logits, axis=0)
            without_first = tl.where(
                expert_lane == route_expert0, -float("inf"), logits
            )
            route_expert1 = tl.argmax(
                without_first, axis=0, tie_break_left=True
            )
            route_logit1 = tl.max(without_first, axis=0)
            if ROUTE_LOGISTIC:
                # For ordered Top-2 logits a >= b, renormalized softmax is a
                # binary logistic.  exp(b-a) is in [0, 1], so this removes the
                # redundant exp(a-a) without overflow.  Compute p1 as e/(1+e)
                # rather than 1-p0: it retains the current path's small-tail
                # accuracy and reaches the identical FP16 checkpoint.
                if USE_FAST_LIBDEVICE:
                    route_ratio = libdevice.fast_expf(
                        route_logit1 - route_logit0
                    )
                else:
                    route_ratio = tl.exp(route_logit1 - route_logit0)
                route_denominator = 1.0 + route_ratio
                route_probability0_h = (
                    1.0 / route_denominator
                ).to(tl.float16)
                route_probability1_h = (
                    route_ratio / route_denominator
                ).to(tl.float16)
            else:
                if USE_FAST_LIBDEVICE:
                    route_exp0 = libdevice.fast_expf(
                        route_logit0 - logits_max
                    )
                    route_exp1 = libdevice.fast_expf(
                        route_logit1 - logits_max
                    )
                else:
                    route_exp0 = tl.exp(route_logit0 - logits_max)
                    route_exp1 = tl.exp(route_logit1 - logits_max)
                route_denominator = route_exp0 + route_exp1
                route_probability0_h = (
                    route_exp0 / route_denominator
                ).to(tl.float16)
                route_probability1_h = (
                    route_exp1 / route_denominator
                ).to(tl.float16)
        else:
            route_denominator = 0.0
            tmp = logits
            for _ in range(TOP_K):
                best = tl.argmax(tmp, axis=0, tie_break_left=True)
                best_logit = tl.max(tmp, axis=0)
                if USE_FAST_LIBDEVICE:
                    route_denominator += libdevice.fast_expf(
                        best_logit - logits_max
                    )
                else:
                    route_denominator += tl.exp(best_logit - logits_max)
                tmp = tl.where(expert_lane == best, -float("inf"), tmp)
    else:
        if USE_FAST_LIBDEVICE:
            logits_exp = libdevice.fast_expf(logits - logits_max)
        else:
            logits_exp = tl.exp(logits - logits_max)
        route_denominator = tl.sum(logits_exp, axis=0)

    x = tl.load(
        hidden_states + token * stride_x_t + hidden_lane * stride_x_h,
        mask=hidden_lane < HIDDEN_SIZE,
        other=0.0,
    ).to(tl.float16)
    accumulated = tl.zeros((BLOCK_H,), dtype=tl.float32)
    candidates = logits

    for route in range(TOP_K):
        if ROUTE_CACHE:
            if route == 0:
                expert = route_expert0
                probability_h = route_probability0_h
            else:
                expert = route_expert1
                probability_h = route_probability1_h
        else:
            expert = tl.argmax(candidates, axis=0, tie_break_left=True)
            selected_logit = tl.max(candidates, axis=0)
            if USE_FAST_LIBDEVICE:
                selected_exp = libdevice.fast_expf(
                    selected_logit - logits_max
                )
            else:
                selected_exp = tl.exp(selected_logit - logits_max)
            probability = selected_exp / route_denominator
            probability_h = probability.to(tl.float16)

        valid_w1 = (inter_lane[:, None] < INTERMEDIATE_SIZE) & (
            hidden_lane[None, :] < HIDDEN_SIZE
        )
        w1_gate = tl.load(
            w1
            + expert * stride_w1_e
            + inter_lane[:, None] * stride_w1_i
            + hidden_lane[None, :] * stride_w1_h,
            mask=valid_w1,
            other=0.0,
        ).to(tl.float16)
        w1_up = tl.load(
            w1
            + expert * stride_w1_e
            + (inter_lane[:, None] + INTERMEDIATE_SIZE) * stride_w1_i
            + hidden_lane[None, :] * stride_w1_h,
            mask=valid_w1,
            other=0.0,
        ).to(tl.float16)

        gate = tl.sum(
            w1_gate.to(tl.float32) * x[None, :].to(tl.float32), axis=1
        ).to(tl.float16)
        up = tl.sum(
            w1_up.to(tl.float32) * x[None, :].to(tl.float32), axis=1
        ).to(tl.float16)
        gate_f = gate.to(tl.float32)
        if USE_FAST_LIBDEVICE:
            gate_exp = libdevice.fast_expf(-gate_f)
        else:
            gate_exp = tl.exp(-gate_f)
        activation = (
            gate_f * (1.0 / (1.0 + gate_exp)) * up.to(tl.float32)
        ).to(tl.float16)

        valid_w2 = (hidden_lane[:, None] < HIDDEN_SIZE) & (
            inter_lane[None, :] < INTERMEDIATE_SIZE
        )
        w2_values = tl.load(
            w2
            + expert * stride_w2_e
            + hidden_lane[:, None] * stride_w2_h
            + inter_lane[None, :] * stride_w2_i,
            mask=valid_w2,
            other=0.0,
        ).to(tl.float16)
        down = tl.sum(
            w2_values.to(tl.float32) * activation[None, :].to(tl.float32),
            axis=1,
        ).to(tl.float16)

        # Match the reference's half-precision expert output and route weight,
        # while retaining an FP32 accumulator for the final two-way reduction.
        contribution = (down * probability_h).to(tl.float16)
        accumulated += contribution.to(tl.float32)
        if not ROUTE_CACHE:
            candidates = tl.where(
                expert_lane == expert, -float("inf"), candidates
            )

    tl.store(
        output + token * stride_o_t + hidden_lane * stride_o_h,
        accumulated,
        mask=hidden_lane < HIDDEN_SIZE,
    )


@triton.jit
def _moe_prepare_kernel(
    hidden_states,
    router_logits,
    w1,
    routed_activation,
    stride_x_t,
    stride_x_h,
    stride_r_t,
    stride_r_e,
    stride_w1_e,
    stride_w1_i,
    stride_w1_h,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    ROUTE_WIDTH: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    """Prepare one FP16 activation row for every token/selected expert.

    Each workspace row is ``[expert_id, probability, activation...]``.  Expert
    IDs are exactly representable in FP16 for the competition's eight experts,
    so metadata and activations share one compact allocation.
    """
    token = tl.program_id(0)
    expert_lane = tl.arange(0, BLOCK_E)
    hidden_lane = tl.arange(0, BLOCK_H)
    inter_lane = tl.arange(0, BLOCK_I)

    logits = tl.load(
        router_logits + token * stride_r_t + expert_lane * stride_r_e,
        mask=expert_lane < NUM_EXPERTS,
        other=-float("inf"),
    ).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    if RENORMALIZE:
        route_denominator = 0.0
        tmp = logits
        for _ in range(TOP_K):
            best = tl.argmax(tmp, axis=0, tie_break_left=True)
            best_logit = tl.max(tmp, axis=0)
            route_denominator += tl.exp(best_logit - logits_max)
            tmp = tl.where(expert_lane == best, -float("inf"), tmp)
    else:
        route_denominator = tl.sum(tl.exp(logits - logits_max), axis=0)

    x = tl.load(
        hidden_states + token * stride_x_t + hidden_lane * stride_x_h,
        mask=hidden_lane < HIDDEN_SIZE,
        other=0.0,
    ).to(tl.float16)
    candidates = logits

    for route in range(TOP_K):
        expert = tl.argmax(candidates, axis=0, tie_break_left=True)
        selected_logit = tl.max(candidates, axis=0)
        probability = (
            tl.exp(selected_logit - logits_max) / route_denominator
        ).to(tl.float16)

        valid_w1 = (inter_lane[:, None] < INTERMEDIATE_SIZE) & (
            hidden_lane[None, :] < HIDDEN_SIZE
        )
        w1_gate = tl.load(
            w1
            + expert * stride_w1_e
            + inter_lane[:, None] * stride_w1_i
            + hidden_lane[None, :] * stride_w1_h,
            mask=valid_w1,
            other=0.0,
        ).to(tl.float16)
        w1_up = tl.load(
            w1
            + expert * stride_w1_e
            + (inter_lane[:, None] + INTERMEDIATE_SIZE) * stride_w1_i
            + hidden_lane[None, :] * stride_w1_h,
            mask=valid_w1,
            other=0.0,
        ).to(tl.float16)

        gate = tl.sum(
            w1_gate.to(tl.float32) * x[None, :].to(tl.float32), axis=1
        ).to(tl.float16)
        up = tl.sum(
            w1_up.to(tl.float32) * x[None, :].to(tl.float32), axis=1
        ).to(tl.float16)
        gate_f = gate.to(tl.float32)
        activation = (
            gate_f * (1.0 / (1.0 + tl.exp(-gate_f))) * up.to(tl.float32)
        ).to(tl.float16)

        route_row = routed_activation + (token * TOP_K + route) * ROUTE_WIDTH
        tl.store(route_row, expert)
        tl.store(route_row + 1, probability)
        tl.store(
            route_row + 2 + inter_lane,
            activation,
            mask=inter_lane < INTERMEDIATE_SIZE,
        )
        candidates = tl.where(expert_lane == expert, -float("inf"), candidates)


@triton.jit
def _moe_finish_kernel(
    routed_activation,
    w2,
    output,
    stride_w2_e,
    stride_w2_h,
    stride_w2_i,
    stride_o_t,
    stride_o_h,
    TOP_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    ROUTE_WIDTH: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Compute an independently owned output-channel tile for one token."""
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden_lane = hidden_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    inter_lane = tl.arange(0, BLOCK_I)
    accumulated = tl.zeros((BLOCK_OH,), dtype=tl.float32)

    for route in range(TOP_K):
        route_row = routed_activation + (token * TOP_K + route) * ROUTE_WIDTH
        expert = tl.load(route_row).to(tl.int32)
        probability = tl.load(route_row + 1).to(tl.float16)
        activation = tl.load(
            route_row + 2 + inter_lane,
            mask=inter_lane < INTERMEDIATE_SIZE,
            other=0.0,
        ).to(tl.float16)

        valid_w2 = (hidden_lane[:, None] < HIDDEN_SIZE) & (
            inter_lane[None, :] < INTERMEDIATE_SIZE
        )
        w2_values = tl.load(
            w2
            + expert * stride_w2_e
            + hidden_lane[:, None] * stride_w2_h
            + inter_lane[None, :] * stride_w2_i,
            mask=valid_w2,
            other=0.0,
        ).to(tl.float16)
        down = tl.sum(
            w2_values.to(tl.float32) * activation[None, :].to(tl.float32),
            axis=1,
        ).to(tl.float16)
        contribution = (down * probability).to(tl.float16)
        accumulated += contribution.to(tl.float32)

    tl.store(
        output + token * stride_o_t + hidden_lane * stride_o_h,
        accumulated,
        mask=hidden_lane < HIDDEN_SIZE,
    )


@triton.jit
def _moe_expert_matmul_kernel(
    hidden_states,
    router_logits,
    w1,
    w2,
    expert_output,
    num_tokens,
    stride_x_t,
    stride_x_h,
    stride_r_t,
    stride_r_e,
    stride_w1_e,
    stride_w1_i,
    stride_w1_h,
    stride_w2_e,
    stride_w2_h,
    stride_w2_i,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    """Matrix-shaped Top-2 expert computation for an expert/token tile.

    Extra work for tokens not routed to this expert is intentional: it turns
    many bandwidth-bound matrix-vector reductions into a small number of
    regular matrix multiplications and reuses each expert's weights across a
    token tile.  Unselected contributions are stored as exact FP16 zeros.
    """
    expert = tl.program_id(0)
    token_block = tl.program_id(1)
    token_lane = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    expert_lane = tl.arange(0, BLOCK_E)
    hidden_lane = tl.arange(0, BLOCK_H)
    inter_lane = tl.arange(0, BLOCK_I)
    valid_token = token_lane < num_tokens

    logits = tl.load(
        router_logits
        + token_lane[:, None] * stride_r_t
        + expert_lane[None, :] * stride_r_e,
        mask=valid_token[:, None] & (expert_lane[None, :] < NUM_EXPERTS),
        other=-float("inf"),
    ).to(tl.float32)
    logits_max = tl.max(logits, axis=1)
    first_expert = tl.argmax(logits, axis=1, tie_break_left=True)
    first_logit = tl.max(logits, axis=1)
    without_first = tl.where(
        expert_lane[None, :] == first_expert[:, None],
        -float("inf"),
        logits,
    )
    second_expert = tl.argmax(without_first, axis=1, tie_break_left=True)
    second_logit = tl.max(without_first, axis=1)

    if RENORMALIZE:
        route_denominator = (
            tl.exp(first_logit - logits_max)
            + tl.exp(second_logit - logits_max)
        )
    else:
        route_denominator = tl.sum(tl.exp(logits - logits_max[:, None]), axis=1)
    selected = (first_expert == expert) | (second_expert == expert)
    selected_logit = tl.where(
        first_expert == expert, first_logit, second_logit
    )
    probability = tl.where(
        selected,
        tl.exp(selected_logit - logits_max) / route_denominator,
        0.0,
    ).to(tl.float16)

    x = tl.load(
        hidden_states
        + token_lane[:, None] * stride_x_t
        + hidden_lane[None, :] * stride_x_h,
        mask=valid_token[:, None] & (hidden_lane[None, :] < HIDDEN_SIZE),
        other=0.0,
    ).to(tl.float16)
    w1_gate = tl.load(
        w1
        + expert * stride_w1_e
        + hidden_lane[:, None] * stride_w1_h
        + inter_lane[None, :] * stride_w1_i,
        mask=(hidden_lane[:, None] < HIDDEN_SIZE)
        & (inter_lane[None, :] < INTERMEDIATE_SIZE),
        other=0.0,
    ).to(tl.float16)
    w1_up = tl.load(
        w1
        + expert * stride_w1_e
        + hidden_lane[:, None] * stride_w1_h
        + (inter_lane[None, :] + INTERMEDIATE_SIZE) * stride_w1_i,
        mask=(hidden_lane[:, None] < HIDDEN_SIZE)
        & (inter_lane[None, :] < INTERMEDIATE_SIZE),
        other=0.0,
    ).to(tl.float16)
    gate = tl.dot(x, w1_gate).to(tl.float16)
    up = tl.dot(x, w1_up).to(tl.float16)
    gate_f = gate.to(tl.float32)
    activation = (
        gate_f * (1.0 / (1.0 + tl.exp(-gate_f))) * up.to(tl.float32)
    ).to(tl.float16)

    w2_values = tl.load(
        w2
        + expert * stride_w2_e
        + inter_lane[:, None] * stride_w2_i
        + hidden_lane[None, :] * stride_w2_h,
        mask=(inter_lane[:, None] < INTERMEDIATE_SIZE)
        & (hidden_lane[None, :] < HIDDEN_SIZE),
        other=0.0,
    ).to(tl.float16)
    down = tl.dot(activation, w2_values).to(tl.float16)
    contribution = (down * probability[:, None]).to(tl.float16)
    contribution = tl.where(selected[:, None], contribution, 0.0)

    tl.store(
        expert_output
        + (expert * num_tokens + token_lane[:, None]) * HIDDEN_SIZE
        + hidden_lane[None, :],
        contribution,
        mask=valid_token[:, None] & (hidden_lane[None, :] < HIDDEN_SIZE),
    )


@triton.jit
def _moe_expert_finalize_kernel(
    expert_output,
    output,
    num_tokens,
    stride_o_t,
    stride_o_h,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_OH: tl.constexpr,
):
    """Reduce the at-most-two nonzero expert contributions per output."""
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    expert_lane = tl.arange(0, BLOCK_E)
    hidden_lane = hidden_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    partial = tl.load(
        expert_output
        + (expert_lane[:, None] * num_tokens + token) * HIDDEN_SIZE
        + hidden_lane[None, :],
        mask=(expert_lane[:, None] < NUM_EXPERTS)
        & (hidden_lane[None, :] < HIDDEN_SIZE),
        other=0.0,
    ).to(tl.float32)
    result = tl.sum(partial, axis=0)
    tl.store(
        output + token * stride_o_t + hidden_lane * stride_o_h,
        result,
        mask=hidden_lane < HIDDEN_SIZE,
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
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.renormalize = bool(renormalize)
        (
            self._num_warps,
            self._implementation,
            self._out_block,
            self._token_block,
        ) = _get_launch_config()
        chip = os.environ.get(
            "KERNELSWIFT_CHIP", "iluvatar_bi150"
        ).strip().lower()
        self._use_compiled_fused = (
            chip == "iluvatar_bi150"
            and self._implementation == "fused"
            and self._num_warps == 8
        )
        self._use_weight_fp16 = (
            chip == "iluvatar_bi150"
            and self._implementation == "fused"
            and os.environ.get(
                "KERNELSWIFT_TASK02_WEIGHT_FP16", "1"
            ).strip()
            == "1"
        )
        self._use_fixed_shape_runner = (
            os.environ.get("KERNELSWIFT_TASK02_FIXED_RUNNER", "1").strip() != "0"
        )
        # Experimental BI150 candidate: for the official renormalized Top-2
        # fused path, retain both selected route ids/logits instead of running
        # the two reductions once for the denominator and again for dispatch.
        # It is deliberately opt-in until measured on the target device.
        self._use_route_cache = (
            self._implementation == "fused"
            and self.renormalize
            and self.top_k == 2
            and os.environ.get(
                "KERNELSWIFT_TASK02_ROUTE_CACHE", "1"
            ).strip()
            == "1"
        )
        # BI150 target testing promoted the stable one-exp logistic route
        # probability for the cached Top-2 path.  Explicit ``two_exp`` is the
        # exact rollback; other chips and non-cached paths retain two-exp.
        route_prob_env = os.environ.get("KERNELSWIFT_TASK02_ROUTE_PROB")
        if route_prob_env is None:
            route_prob = (
                "logistic"
                if chip == "iluvatar_bi150" and self._use_route_cache
                else "two_exp"
            )
        else:
            route_prob = route_prob_env.strip().lower()
        self._use_route_logistic = (
            chip == "iluvatar_bi150"
            and self._use_route_cache
            and route_prob == "logistic"
        )
        self._use_fast_libdevice = (
            chip == "iluvatar_bi150"
            and self._implementation == "fused"
            and os.environ.get(
                "KERNELSWIFT_TASK02_MATH", "fast_libdevice"
            ).strip().lower()
            == "fast_libdevice"
        )
        self._fixed_fused_runner = None
        self._fixed_fused_runner_device = None
        # Cache only CompiledKernel launchers.  Outputs are always freshly
        # allocated in forward and are never retained here.
        self._compiled_fused_runners = {}
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.float32,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.float32,
            )
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        if self._use_weight_fp16:
            # Keep the public parameter names/state-dict keys unchanged.  A
            # normal load_state_dict from the FP32 reference now rounds once
            # while copying into FP16 storage, exactly where the fused kernel
            # previously rounded each FP32 load with ``.to(tl.float16)``.
            self.w1 = nn.Parameter(
                self.w1.detach().to(torch.float16),
                requires_grad=self.w1.requires_grad,
            )
            self.w2 = nn.Parameter(
                self.w2.detach().to(torch.float16),
                requires_grad=self.w2.requires_grad,
            )

        graph_value = os.environ.get("KERNELSWIFT_TASK02_GRAPH_REPLAY")
        if graph_value is None:
                graph_value = os.environ.get("KERNELSWIFT_BI150_GRAPH_REPLAY", "1")
        self._use_graph_replay = (
            chip == "iluvatar_bi150"
            and graph_value.strip().lower() in ("1", "true", "yes")
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_output = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_graph_replay or hidden_states.device.type == "cpu":
            return self._forward_impl(hidden_states, router_logits)

        graph_inputs = self._graph_inputs
        if (
            self._graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is hidden_states
            and graph_inputs[1] is router_logits
            and graph_inputs[2] is self.w1
            and graph_inputs[3] is self.w2
        ):
            self._graph.replay()
            return self._graph_output

        self._forward_impl(hidden_states, router_logits)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward_impl(hidden_states, router_logits)
        self._graph = graph
        self._graph_inputs = (hidden_states, router_logits, self.w1, self.w2)
        self._graph_output = output
        graph.replay()
        return output

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [T, hidden_size]")
        if router_logits.shape != (hidden_states.shape[0], self.num_experts):
            raise ValueError("router_logits must have shape [T, num_experts]")
        if not (0 < self.top_k <= self.num_experts):
            raise ValueError("top_k must be in [1, num_experts]")
        if hidden_states.dtype != torch.float16:
            raise TypeError("the competition kernel expects float16 hidden states")

        num_tokens = hidden_states.shape[0]
        output = torch.empty_like(hidden_states)
        block_e = triton.next_power_of_2(self.num_experts)
        block_h = triton.next_power_of_2(self.hidden_size)
        block_i = triton.next_power_of_2(self.intermediate_size)

        if self._implementation == "expert" and self.top_k == 2:
            expert_output = torch.empty(
                (self.num_experts, num_tokens, self.hidden_size),
                dtype=torch.float16,
                device=hidden_states.device,
            )
            token_blocks = triton.cdiv(num_tokens, self._token_block)
            _moe_expert_matmul_kernel[(self.num_experts, token_blocks)](
                hidden_states,
                router_logits,
                self.w1,
                self.w2,
                expert_output,
                num_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                router_logits.stride(0),
                router_logits.stride(1),
                self.w1.stride(0),
                self.w1.stride(1),
                self.w1.stride(2),
                self.w2.stride(0),
                self.w2.stride(1),
                self.w2.stride(2),
                NUM_EXPERTS=self.num_experts,
                HIDDEN_SIZE=self.hidden_size,
                INTERMEDIATE_SIZE=self.intermediate_size,
                BLOCK_T=self._token_block,
                BLOCK_E=block_e,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                RENORMALIZE=self.renormalize,
                num_warps=self._num_warps,
            )
            output_blocks = triton.cdiv(self.hidden_size, self._out_block)
            _moe_expert_finalize_kernel[(num_tokens, output_blocks)](
                expert_output,
                output,
                num_tokens,
                output.stride(0),
                output.stride(1),
                NUM_EXPERTS=self.num_experts,
                HIDDEN_SIZE=self.hidden_size,
                BLOCK_E=block_e,
                BLOCK_OH=self._out_block,
                num_warps=self._num_warps,
            )
            return output

        if self._implementation == "split":
            route_width = self.intermediate_size + 2
            routed_activation = torch.empty(
                (num_tokens, self.top_k, route_width),
                dtype=torch.float16,
                device=hidden_states.device,
            )
            _moe_prepare_kernel[(num_tokens,)](
                hidden_states,
                router_logits,
                self.w1,
                routed_activation,
                hidden_states.stride(0),
                hidden_states.stride(1),
                router_logits.stride(0),
                router_logits.stride(1),
                self.w1.stride(0),
                self.w1.stride(1),
                self.w1.stride(2),
                NUM_EXPERTS=self.num_experts,
                TOP_K=self.top_k,
                HIDDEN_SIZE=self.hidden_size,
                INTERMEDIATE_SIZE=self.intermediate_size,
                ROUTE_WIDTH=route_width,
                BLOCK_E=block_e,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                RENORMALIZE=self.renormalize,
                num_warps=self._num_warps,
            )
            output_blocks = triton.cdiv(self.hidden_size, self._out_block)
            _moe_finish_kernel[(num_tokens, output_blocks)](
                routed_activation,
                self.w2,
                output,
                self.w2.stride(0),
                self.w2.stride(1),
                self.w2.stride(2),
                output.stride(0),
                output.stride(1),
                TOP_K=self.top_k,
                HIDDEN_SIZE=self.hidden_size,
                INTERMEDIATE_SIZE=self.intermediate_size,
                ROUTE_WIDTH=route_width,
                BLOCK_OH=self._out_block,
                BLOCK_I=block_i,
                num_warps=self._num_warps,
            )
            return output

        # The published benchmark has one fixed, dense layout.  On CoreX the
        # Python work needed to rebuild the complete generic cache key is a
        # meaningful fraction of this very small kernel.  Retain the generic
        # path below for every other legal layout, but keep a single executable
        # launcher for the exact official case.  Only code is cached: ``output``
        # is newly allocated and fully overwritten on every call.
        fixed_layout = (
            self._use_compiled_fused
            and self._use_fixed_shape_runner
            and num_tokens == 83
            and self.num_experts == 8
            and self.top_k == 2
            and self.hidden_size == 128
            and self.intermediate_size == 64
            and hidden_states.is_contiguous()
            and router_logits.is_contiguous()
            and self.w1.is_contiguous()
            and self.w2.is_contiguous()
        )
        if fixed_layout and hasattr(_fused_moe_kernel, "warmup"):
            if (
                self._fixed_fused_runner is None
                or self._fixed_fused_runner_device != hidden_states.device
            ):
                compiled_grid = (83, 1, 1)
                compiled = _fused_moe_kernel.warmup(
                    hidden_states,
                    router_logits,
                    self.w1,
                    self.w2,
                    output,
                    hidden_states.stride(0),
                    hidden_states.stride(1),
                    router_logits.stride(0),
                    router_logits.stride(1),
                    self.w1.stride(0),
                    self.w1.stride(1),
                    self.w1.stride(2),
                    self.w2.stride(0),
                    self.w2.stride(1),
                    self.w2.stride(2),
                    output.stride(0),
                    output.stride(1),
                    NUM_EXPERTS=8,
                    TOP_K=2,
                    HIDDEN_SIZE=128,
                    INTERMEDIATE_SIZE=64,
                    BLOCK_E=8,
                    BLOCK_H=128,
                    BLOCK_I=64,
                    RENORMALIZE=self.renormalize,
                    ROUTE_CACHE=self._use_route_cache,
                    ROUTE_LOGISTIC=self._use_route_logistic,
                    USE_FAST_LIBDEVICE=self._use_fast_libdevice,
                    grid=compiled_grid,
                    num_warps=8,
                )
                self._fixed_fused_runner = compiled[compiled_grid]
                self._fixed_fused_runner_device = hidden_states.device
            self._fixed_fused_runner(
                hidden_states, router_logits, self.w1, self.w2, output
            )
            return output

        fused_args = (
            hidden_states,
            router_logits,
            self.w1,
            self.w2,
            output,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_logits.stride(0),
            router_logits.stride(1),
            self.w1.stride(0),
            self.w1.stride(1),
            self.w1.stride(2),
            self.w2.stride(0),
            self.w2.stride(1),
            self.w2.stride(2),
            output.stride(0),
            output.stride(1),
        )
        if self._use_compiled_fused and hasattr(_fused_moe_kernel, "warmup"):
            compiled_grid = (num_tokens, 1, 1)
            runner_key = (
                self._implementation,
                self._num_warps,
                self._out_block,
                self._token_block,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.renormalize,
                block_e,
                block_h,
                block_i,
                self._use_route_cache,
                self._use_route_logistic,
                self._use_fast_libdevice,
                compiled_grid,
                _tensor_layout_key(hidden_states),
                _tensor_layout_key(router_logits),
                _tensor_layout_key(self.w1),
                _tensor_layout_key(self.w2),
                _tensor_layout_key(output),
            )
            runner = self._compiled_fused_runners.get(runner_key)
            if runner is None:
                compiled = _fused_moe_kernel.warmup(
                    *fused_args,
                    NUM_EXPERTS=self.num_experts,
                    TOP_K=self.top_k,
                    HIDDEN_SIZE=self.hidden_size,
                    INTERMEDIATE_SIZE=self.intermediate_size,
                    BLOCK_E=block_e,
                    BLOCK_H=block_h,
                    BLOCK_I=block_i,
                    RENORMALIZE=self.renormalize,
                    ROUTE_CACHE=self._use_route_cache,
                    ROUTE_LOGISTIC=self._use_route_logistic,
                    USE_FAST_LIBDEVICE=self._use_fast_libdevice,
                    grid=compiled_grid,
                    num_warps=self._num_warps,
                )
                runner = compiled[compiled_grid]
                self._compiled_fused_runners[runner_key] = runner
            # Tensor layout is already part of ``runner_key``.  Specializing
            # the twelve strides reduces the hot-path CoreX launch signature
            # from seventeen arguments to the five pointers below.
            runner(*fused_args[:5])
            return output

        _fused_moe_kernel[(num_tokens,)](
            *fused_args,
            NUM_EXPERTS=self.num_experts,
            TOP_K=self.top_k,
            HIDDEN_SIZE=self.hidden_size,
            INTERMEDIATE_SIZE=self.intermediate_size,
            BLOCK_E=block_e,
            BLOCK_H=block_h,
            BLOCK_I=block_i,
            RENORMALIZE=self.renormalize,
            ROUTE_CACHE=self._use_route_cache,
            ROUTE_LOGISTIC=self._use_route_logistic,
            USE_FAST_LIBDEVICE=self._use_fast_libdevice,
            num_warps=self._num_warps,
        )
        return output


class Model(ModelNew):
    """Compatibility name for runners that import ``Model``."""


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device="cuda"
    )
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda"
    )
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]
