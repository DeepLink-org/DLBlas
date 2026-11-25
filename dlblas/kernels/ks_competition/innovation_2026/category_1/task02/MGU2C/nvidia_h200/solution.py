"""A one-launch Triton implementation of the small routed SwiGLU MoE."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from cuda.bindings import runtime as _cuda_runtime


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
    "nvidia_h200": (8,),
}


_VALID_WARPS = (1, 2, 4, 8)
_VALID_IMPLEMENTATIONS = (
    "fused",
    "fused_top2",
    "fused_gather",
    "fused_bitonic",
    "route_parallel",
    "recompute",
    "split",
    "expert",
)
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


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
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


@triton.jit
def _fused_moe_kernel(
    hidden_states,
    router_logits,
    w1,
    w2,
    output,
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
    stride_o_t,
    stride_o_h,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    PRESELECT_TOP2: tl.constexpr,
    GATHER_TOP2_LOGITS: tl.constexpr,
    USE_BITONIC_TOP2: tl.constexpr,
    PARALLEL_ROUTES: tl.constexpr,
    ROUTES_PER_PROGRAM: tl.constexpr,
    FIXED_CONTIGUOUS_LAYOUT: tl.constexpr,
):
    token = tl.program_id(0)
    if PARALLEL_ROUTES:
        route_program = tl.program_id(1)
    else:
        route_program = 0
    expert_lane = tl.arange(0, BLOCK_E)
    hidden_lane = tl.arange(0, BLOCK_H)
    inter_lane = tl.arange(0, BLOCK_I)

    if FIXED_CONTIGUOUS_LAYOUT:
        router_offsets = token * NUM_EXPERTS + expert_lane
    else:
        router_offsets = token * stride_r_t + expert_lane * stride_r_e
    logits = tl.load(
        router_logits + router_offsets,
        mask=expert_lane < NUM_EXPERTS,
        other=-float("inf"),
    ).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    # ``fused_top2`` selects both routes once and reuses those scalar SSA
    # values for the denominator and expert loop.  The default fused variant
    # deliberately retains the previously validated two-pass selection.
    route_lane = tl.arange(0, TOP_K)
    if USE_BITONIC_TOP2:
        router_bits = logits.to(tl.uint32, bitcast=True)
        router_bits = tl.where(logits == 0.0, 0, router_bits)
        router_negative = (router_bits & 0x80000000) != 0
        router_order = tl.where(
            router_negative,
            router_bits ^ 0xFFFFFFFF,
            router_bits ^ 0x80000000,
        )
        router_tie = BLOCK_E - 1 - expert_lane
        router_keys = (
            router_order.to(tl.uint64) << 32
        ) | router_tie.to(tl.uint64)
        selected_keys = tl.topk(router_keys, TOP_K)
        selected_experts = BLOCK_E - 1 - (
            selected_keys & 0xFFFFFFFF
        ).to(tl.int32)
        selected_order = (selected_keys >> 32).to(tl.uint32)
        selected_positive = (selected_order & 0x80000000) != 0
        selected_bits = tl.where(
            selected_positive,
            selected_order ^ 0x80000000,
            selected_order ^ 0xFFFFFFFF,
        )
        selected_logits = selected_bits.to(tl.float32, bitcast=True)
        if RENORMALIZE:
            route_denominator = tl.sum(
                tl.exp(selected_logits - logits_max), axis=0
            )
        else:
            route_denominator = tl.sum(
                tl.exp(logits - logits_max), axis=0
            )
    elif PRESELECT_TOP2 or GATHER_TOP2_LOGITS or PARALLEL_ROUTES:
        first_expert = tl.argmax(logits, axis=0, tie_break_left=True)
        if GATHER_TOP2_LOGITS:
            if FIXED_CONTIGUOUS_LAYOUT:
                first_offset = token * NUM_EXPERTS + first_expert
            else:
                first_offset = token * stride_r_t + first_expert * stride_r_e
            first_logit = tl.load(router_logits + first_offset).to(tl.float32)
        else:
            first_logit = tl.max(logits, axis=0)
        without_first = tl.where(
            expert_lane == first_expert, -float("inf"), logits
        )
        second_expert = tl.argmax(
            without_first, axis=0, tie_break_left=True
        )
        if GATHER_TOP2_LOGITS:
            if FIXED_CONTIGUOUS_LAYOUT:
                second_offset = token * NUM_EXPERTS + second_expert
            else:
                second_offset = token * stride_r_t + second_expert * stride_r_e
            second_logit = tl.load(router_logits + second_offset).to(tl.float32)
        else:
            second_logit = tl.max(without_first, axis=0)
        if RENORMALIZE:
            route_denominator = (
                tl.exp(first_logit - logits_max)
                + tl.exp(second_logit - logits_max)
            )
        else:
            route_denominator = tl.sum(
                tl.exp(logits - logits_max), axis=0
            )
    else:
        # Softmax is strictly increasing, so Top-K is performed on logits.
        # With renormalization the full-router denominator cancels and only
        # TOP_K exponentials are needed for the final stable softmax.
        if RENORMALIZE:
            route_denominator = 0.0
            tmp = logits
            for _ in range(TOP_K):
                best = tl.argmax(tmp, axis=0, tie_break_left=True)
                best_logit = tl.max(tmp, axis=0)
                route_denominator += tl.exp(best_logit - logits_max)
                tmp = tl.where(
                    expert_lane == best, -float("inf"), tmp
                )
        else:
            logits_exp = tl.exp(logits - logits_max)
            route_denominator = tl.sum(logits_exp, axis=0)

    if FIXED_CONTIGUOUS_LAYOUT:
        hidden_offsets = token * HIDDEN_SIZE + hidden_lane
    else:
        hidden_offsets = token * stride_x_t + hidden_lane * stride_x_h
    x = tl.load(
        hidden_states + hidden_offsets,
        mask=hidden_lane < HIDDEN_SIZE,
        other=0.0,
    ).to(tl.float16)
    accumulated = tl.zeros((BLOCK_H,), dtype=tl.float32)
    candidates = logits

    for route in range(ROUTES_PER_PROGRAM):
        if PARALLEL_ROUTES:
            # Each CTA owns exactly one of the two selected experts.  Scalar
            # selects avoid a dynamic branch while expanding the official
            # grid from 83 to 166 CTAs.
            expert = tl.where(
                route_program == 0, first_expert, second_expert
            )
            selected_logit = tl.where(
                route_program == 0, first_logit, second_logit
            )
        elif USE_BITONIC_TOP2:
            expert = tl.sum(
                tl.where(route_lane == route, selected_experts, 0), axis=0
            )
            selected_logit = tl.sum(
                tl.where(route_lane == route, selected_logits, 0.0), axis=0
            )
        elif PRESELECT_TOP2 or GATHER_TOP2_LOGITS:
            if route == 0:
                expert = first_expert
                selected_logit = first_logit
            else:
                expert = second_expert
                selected_logit = second_logit
        else:
            expert = tl.argmax(candidates, axis=0, tie_break_left=True)
            selected_logit = tl.max(candidates, axis=0)
        probability = tl.exp(selected_logit - logits_max) / route_denominator
        probability_h = probability.to(tl.float16)

        valid_w1 = (inter_lane[:, None] < INTERMEDIATE_SIZE) & (
            hidden_lane[None, :] < HIDDEN_SIZE
        )
        if FIXED_CONTIGUOUS_LAYOUT:
            w1_gate_offsets = (
                expert * (2 * INTERMEDIATE_SIZE * HIDDEN_SIZE)
                + inter_lane[:, None] * HIDDEN_SIZE
                + hidden_lane[None, :]
            )
            w1_up_offsets = w1_gate_offsets + INTERMEDIATE_SIZE * HIDDEN_SIZE
        else:
            w1_gate_offsets = (
                expert * stride_w1_e
                + inter_lane[:, None] * stride_w1_i
                + hidden_lane[None, :] * stride_w1_h
            )
            w1_up_offsets = w1_gate_offsets + INTERMEDIATE_SIZE * stride_w1_i
        w1_gate = tl.load(
            w1 + w1_gate_offsets,
            mask=valid_w1,
            other=0.0,
        ).to(tl.float16)
        w1_up = tl.load(
            w1 + w1_up_offsets,
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

        valid_w2 = (hidden_lane[:, None] < HIDDEN_SIZE) & (
            inter_lane[None, :] < INTERMEDIATE_SIZE
        )
        if FIXED_CONTIGUOUS_LAYOUT:
            w2_offsets = (
                expert * (HIDDEN_SIZE * INTERMEDIATE_SIZE)
                + hidden_lane[:, None] * INTERMEDIATE_SIZE
                + inter_lane[None, :]
            )
        else:
            w2_offsets = (
                expert * stride_w2_e
                + hidden_lane[:, None] * stride_w2_h
                + inter_lane[None, :] * stride_w2_i
            )
        w2_values = tl.load(
            w2 + w2_offsets,
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
        if (
            not PRESELECT_TOP2
            and not GATHER_TOP2_LOGITS
            and not PARALLEL_ROUTES
        ):
            candidates = tl.where(
                expert_lane == expert, -float("inf"), candidates
            )

    if FIXED_CONTIGUOUS_LAYOUT:
        output_offsets = token * HIDDEN_SIZE + hidden_lane
    else:
        output_offsets = token * stride_o_t + hidden_lane * stride_o_h
    if PARALLEL_ROUTES:
        # ``output`` is zero-initialised by the guarded caller.  Each expert
        # contribution is already rounded to FP16 above, so the only two-way
        # FP16 addition matches the reference's half output reduction.
        tl.atomic_add(
            output + output_offsets,
            accumulated.to(tl.float16),
            mask=hidden_lane < HIDDEN_SIZE,
        )
    else:
        tl.store(
            output + output_offsets,
            accumulated,
            mask=hidden_lane < HIDDEN_SIZE,
        )


@triton.jit
def _fused_moe_recompute_kernel(
    hidden_states,
    router_logits,
    w1,
    w2,
    output,
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
    stride_o_t,
    stride_o_h,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    FIXED_CONTIGUOUS_LAYOUT: tl.constexpr,
):
    """One-launch Top-2 MoE with independently owned output-channel tiles.

    The established fused kernel launches one program per token, only 83
    programs for the official shape.  This opt-in candidate launches two or
    four programs per token and keeps every output store race-free.  Each tile
    deliberately recomputes the small gate/up projection; that extra work can
    be cheaper on H200 than leaving a third of its 132 SMs idle or introducing
    an activation workspace and a second launch.
    """
    token = tl.program_id(0)
    output_block = tl.program_id(1)
    expert_lane = tl.arange(0, BLOCK_E)
    hidden_lane = tl.arange(0, BLOCK_H)
    inter_lane = tl.arange(0, BLOCK_I)
    output_lane = output_block * BLOCK_OH + tl.arange(0, BLOCK_OH)

    if FIXED_CONTIGUOUS_LAYOUT:
        router_offsets = token * NUM_EXPERTS + expert_lane
    else:
        router_offsets = token * stride_r_t + expert_lane * stride_r_e
    logits = tl.load(
        router_logits + router_offsets,
        mask=expert_lane < NUM_EXPERTS,
        other=-float("inf"),
    ).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    first_expert = tl.argmax(logits, axis=0, tie_break_left=True)
    if FIXED_CONTIGUOUS_LAYOUT:
        first_offset = token * NUM_EXPERTS + first_expert
    else:
        first_offset = token * stride_r_t + first_expert * stride_r_e
    first_logit = tl.load(router_logits + first_offset).to(tl.float32)
    without_first = tl.where(
        expert_lane == first_expert, -float("inf"), logits
    )
    second_expert = tl.argmax(without_first, axis=0, tie_break_left=True)
    if FIXED_CONTIGUOUS_LAYOUT:
        second_offset = token * NUM_EXPERTS + second_expert
    else:
        second_offset = token * stride_r_t + second_expert * stride_r_e
    second_logit = tl.load(router_logits + second_offset).to(tl.float32)
    if RENORMALIZE:
        route_denominator = (
            tl.exp(first_logit - logits_max)
            + tl.exp(second_logit - logits_max)
        )
    else:
        route_denominator = tl.sum(tl.exp(logits - logits_max), axis=0)

    if FIXED_CONTIGUOUS_LAYOUT:
        hidden_offsets = token * HIDDEN_SIZE + hidden_lane
    else:
        hidden_offsets = token * stride_x_t + hidden_lane * stride_x_h
    x = tl.load(
        hidden_states + hidden_offsets,
        mask=hidden_lane < HIDDEN_SIZE,
        other=0.0,
    ).to(tl.float16)
    accumulated = tl.zeros((BLOCK_OH,), dtype=tl.float32)

    for route in range(2):
        if route == 0:
            expert = first_expert
            selected_logit = first_logit
        else:
            expert = second_expert
            selected_logit = second_logit
        probability_h = (
            tl.exp(selected_logit - logits_max) / route_denominator
        ).to(tl.float16)

        valid_w1 = (inter_lane[:, None] < INTERMEDIATE_SIZE) & (
            hidden_lane[None, :] < HIDDEN_SIZE
        )
        if FIXED_CONTIGUOUS_LAYOUT:
            w1_gate_offsets = (
                expert * (2 * INTERMEDIATE_SIZE * HIDDEN_SIZE)
                + inter_lane[:, None] * HIDDEN_SIZE
                + hidden_lane[None, :]
            )
            w1_up_offsets = w1_gate_offsets + INTERMEDIATE_SIZE * HIDDEN_SIZE
        else:
            w1_gate_offsets = (
                expert * stride_w1_e
                + inter_lane[:, None] * stride_w1_i
                + hidden_lane[None, :] * stride_w1_h
            )
            w1_up_offsets = w1_gate_offsets + INTERMEDIATE_SIZE * stride_w1_i
        w1_gate = tl.load(
            w1 + w1_gate_offsets, mask=valid_w1, other=0.0
        ).to(tl.float16)
        w1_up = tl.load(
            w1 + w1_up_offsets, mask=valid_w1, other=0.0
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

        valid_w2 = (output_lane[:, None] < HIDDEN_SIZE) & (
            inter_lane[None, :] < INTERMEDIATE_SIZE
        )
        if FIXED_CONTIGUOUS_LAYOUT:
            w2_offsets = (
                expert * (HIDDEN_SIZE * INTERMEDIATE_SIZE)
                + output_lane[:, None] * INTERMEDIATE_SIZE
                + inter_lane[None, :]
            )
        else:
            w2_offsets = (
                expert * stride_w2_e
                + output_lane[:, None] * stride_w2_h
                + inter_lane[None, :] * stride_w2_i
            )
        w2_values = tl.load(
            w2 + w2_offsets, mask=valid_w2, other=0.0
        ).to(tl.float16)
        down = tl.sum(
            w2_values.to(tl.float32) * activation[None, :].to(tl.float32),
            axis=1,
        ).to(tl.float16)
        accumulated += (down * probability_h).to(tl.float16).to(tl.float32)

    if FIXED_CONTIGUOUS_LAYOUT:
        output_offsets = token * HIDDEN_SIZE + output_lane
    else:
        output_offsets = token * stride_o_t + output_lane * stride_o_h
    tl.store(
        output + output_offsets,
        accumulated,
        mask=output_lane < HIDDEN_SIZE,
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
        self._prepack_fp16 = _env_flag("KERNELSWIFT_TASK02_PREPACK_FP16")
        self._fixed_contiguous_layout = (
            os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
            == "nvidia_h200"
            and _env_flag("KERNELSWIFT_TASK02_FIXED_LAYOUT")
        )
        chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
        launch_mode = os.environ.get(
            "KERNELSWIFT_TASK02_LAUNCH_MODE",
            "direct" if chip == "nvidia_h200" else "jit",
        ).strip().lower()
        if launch_mode not in ("jit", "direct"):
            raise ValueError("Task02 LAUNCH_MODE must be jit or direct")
        self._direct_launch_requested = (
            launch_mode == "direct"
            and chip == "nvidia_h200"
            and getattr(triton, "__version__", "") == "3.6.0"
        )
        self._direct_runner = None
        self._direct_device = None
        self.register_buffer("_w1_fp16", None, persistent=False)
        self.register_buffer("_w2_fp16", None, persistent=False)
        self._refresh_prepacked_weights()

        graph_key = os.environ.get("KERNELSWIFT_TASK02_GRAPH_REPLAY")
        if graph_key is None:
            graph_key = os.environ.get("KERNELSWIFT_H200_GRAPH_REPLAY", "1")
        graph_key = graph_key.strip().lower()
        if graph_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task02 GRAPH_REPLAY must be 0 or 1")
        self._use_graph_replay = (
            chip == "nvidia_h200"
            and graph_key in ("1", "true", "yes")
            and torch.cuda.is_available()
        )
        raw_graph_key = os.environ.get(
            "KERNELSWIFT_TASK02_RAW_GRAPH_LAUNCH", "1"
        ).strip().lower()
        if raw_graph_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task02 RAW_GRAPH_LAUNCH must be 0 or 1")
        self._use_raw_graph_launch = (
            self._use_graph_replay
            and raw_graph_key in ("1", "true", "yes")
            and getattr(triton, "__version__", "") == "3.6.0"
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_output = None
        self._raw_graph_exec = None
        self._raw_graph_device_index = None

    def _refresh_prepacked_weights(self) -> None:
        if not self._prepack_fp16:
            self._w1_fp16 = None
            self._w2_fp16 = None
            return
        # Keep the official FP32 Parameters untouched.  These derived buffers
        # are excluded from state_dict and follow the module through .to().
        self._w1_fp16 = self.w1.detach().to(dtype=torch.float16)
        self._w2_fp16 = self.w2.detach().to(dtype=torch.float16)

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_prepacked_weights()
        self._graph = None
        self._graph_inputs = None
        self._graph_output = None
        self._raw_graph_exec = None
        self._raw_graph_device_index = None
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        w1 = self._w1_fp16 if self._prepack_fp16 else self.w1
        w2 = self._w2_fp16 if self._prepack_fp16 else self.w2
        graph = self._graph
        graph_inputs = self._graph_inputs
        if (
            graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is hidden_states
            and graph_inputs[1] is router_logits
            and graph_inputs[2] is w1
            and graph_inputs[3] is w2
        ):
            raw_graph_exec = self._raw_graph_exec
            if raw_graph_exec is None:
                graph.replay()
            else:
                stream = _cuda_runtime.cudaStream_t(
                    torch._C._cuda_getCurrentRawStream(
                        self._raw_graph_device_index
                    )
                )
                status = _cuda_runtime.cudaGraphLaunch(
                    raw_graph_exec, stream
                )[0]
                if status != _cuda_runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaGraphLaunch failed: {status}")
            return self._graph_output

        official = (
            self._use_graph_replay
            and self._direct_launch_requested
            and self._implementation == "fused"
            and self.num_experts == 8
            and self.top_k == 2
            and self.hidden_size == 128
            and self.intermediate_size == 64
            and hidden_states.shape == (83, 128)
            and router_logits.shape == (83, 8)
            and hidden_states.dtype == torch.float16
            and router_logits.dtype == torch.float32
            and hidden_states.device == router_logits.device == w1.device == w2.device
            and hidden_states.device.type == "cuda"
            and hidden_states.is_contiguous()
            and router_logits.is_contiguous()
            and w1.is_contiguous()
            and w2.is_contiguous()
        )
        if not official:
            return self._forward_impl(hidden_states, router_logits)

        self._forward_impl(hidden_states, router_logits)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward_impl(hidden_states, router_logits)
        self._graph = graph
        self._graph_inputs = (hidden_states, router_logits, w1, w2)
        self._graph_output = output
        if self._use_raw_graph_launch:
            self._raw_graph_exec = _cuda_runtime.cudaGraphExec_t(
                graph.raw_cuda_graph_exec()
            )
            self._raw_graph_device_index = hidden_states.device.index
        else:
            self._raw_graph_exec = None
            self._raw_graph_device_index = None
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

        w1 = self._w1_fp16 if self._prepack_fp16 else self.w1
        w2 = self._w2_fp16 if self._prepack_fp16 else self.w2
        num_tokens = hidden_states.shape[0]
        route_parallel = (
            self._implementation == "route_parallel"
            and num_tokens == 83
            and self.num_experts == 8
            and self.top_k == 2
            and self.hidden_size == 128
            and self.intermediate_size == 64
            and hidden_states.shape == (83, 128)
            and router_logits.shape == (83, 8)
            and hidden_states.dtype == torch.float16
            and router_logits.dtype == torch.float32
            and hidden_states.is_contiguous()
            and router_logits.is_contiguous()
            and w1.is_contiguous()
            and w2.is_contiguous()
        )
        output = (
            torch.zeros_like(hidden_states)
            if route_parallel
            else torch.empty_like(hidden_states)
        )
        block_e = triton.next_power_of_2(self.num_experts)
        block_h = triton.next_power_of_2(self.hidden_size)
        block_i = triton.next_power_of_2(self.intermediate_size)
        fixed_contiguous_layout = (
            self._fixed_contiguous_layout
            and self.num_experts == 8
            and self.top_k == 2
            and self.hidden_size == 128
            and self.intermediate_size == 64
            and hidden_states.is_contiguous()
            and router_logits.is_contiguous()
            and w1.is_contiguous()
            and w2.is_contiguous()
            and output.is_contiguous()
        )

        if self._implementation == "recompute" and self.top_k == 2:
            output_blocks = triton.cdiv(self.hidden_size, self._out_block)
            _fused_moe_recompute_kernel[(num_tokens, output_blocks)](
                hidden_states,
                router_logits,
                w1,
                w2,
                output,
                hidden_states.stride(0),
                hidden_states.stride(1),
                router_logits.stride(0),
                router_logits.stride(1),
                w1.stride(0),
                w1.stride(1),
                w1.stride(2),
                w2.stride(0),
                w2.stride(1),
                w2.stride(2),
                output.stride(0),
                output.stride(1),
                NUM_EXPERTS=self.num_experts,
                HIDDEN_SIZE=self.hidden_size,
                INTERMEDIATE_SIZE=self.intermediate_size,
                BLOCK_E=block_e,
                BLOCK_H=block_h,
                BLOCK_I=block_i,
                BLOCK_OH=self._out_block,
                RENORMALIZE=self.renormalize,
                FIXED_CONTIGUOUS_LAYOUT=fixed_contiguous_layout,
                num_warps=self._num_warps,
            )
            return output

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
                w1,
                w2,
                expert_output,
                num_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                router_logits.stride(0),
                router_logits.stride(1),
                w1.stride(0),
                w1.stride(1),
                w1.stride(2),
                w2.stride(0),
                w2.stride(1),
                w2.stride(2),
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
                w1,
                routed_activation,
                hidden_states.stride(0),
                hidden_states.stride(1),
                router_logits.stride(0),
                router_logits.stride(1),
                w1.stride(0),
                w1.stride(1),
                w1.stride(2),
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
                w2,
                output,
                w2.stride(0),
                w2.stride(1),
                w2.stride(2),
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

        fused_grid = (num_tokens, 2) if route_parallel else (num_tokens,)
        preselect_top2 = (
            (self._implementation == "fused_top2" and self.top_k == 2)
            or route_parallel
        )
        gather_top2 = (
            self._implementation == "fused_gather" and self.top_k == 2
        )
        bitonic_top2 = (
            self._implementation == "fused_bitonic" and self.top_k == 2
        )
        routes_per_program = 1 if route_parallel else self.top_k
        use_direct_launch = (
            self._direct_launch_requested
            and num_tokens == 83
            and self.num_experts == 8
            and self.top_k == 2
            and self.hidden_size == 128
            and self.intermediate_size == 64
            and hidden_states.shape == (83, 128)
            and router_logits.shape == (83, 8)
            and hidden_states.dtype == torch.float16
            and router_logits.dtype == torch.float32
            and hidden_states.device == router_logits.device
            and hidden_states.is_contiguous()
            and router_logits.is_contiguous()
            and w1.is_contiguous()
            and w2.is_contiguous()
            and output.is_contiguous()
        )
        if use_direct_launch:
            signature_args = (
                hidden_states,
                router_logits,
                w1,
                w2,
                output,
                hidden_states.stride(0),
                hidden_states.stride(1),
                router_logits.stride(0),
                router_logits.stride(1),
                w1.stride(0),
                w1.stride(1),
                w1.stride(2),
                w2.stride(0),
                w2.stride(1),
                w2.stride(2),
                output.stride(0),
                output.stride(1),
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                block_e,
                block_h,
                block_i,
                self.renormalize,
                preselect_top2,
                gather_top2,
                bitonic_top2,
                route_parallel,
                routes_per_program,
                fixed_contiguous_layout,
            )
            if (
                self._direct_runner is not None
                and hidden_states.device == self._direct_device
            ):
                self._direct_runner(*signature_args)
            else:
                compiled = _fused_moe_kernel[fused_grid](
                    *signature_args, num_warps=self._num_warps
                )
                self._direct_runner = compiled[
                    (num_tokens, 2 if route_parallel else 1, 1)
                ]
                self._direct_device = hidden_states.device
            return output
        _fused_moe_kernel[fused_grid](
            hidden_states,
            router_logits,
            w1,
            w2,
            output,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_logits.stride(0),
            router_logits.stride(1),
            w1.stride(0),
            w1.stride(1),
            w1.stride(2),
            w2.stride(0),
            w2.stride(1),
            w2.stride(2),
            output.stride(0),
            output.stride(1),
            NUM_EXPERTS=self.num_experts,
            TOP_K=self.top_k,
            HIDDEN_SIZE=self.hidden_size,
            INTERMEDIATE_SIZE=self.intermediate_size,
            BLOCK_E=block_e,
            BLOCK_H=block_h,
            BLOCK_I=block_i,
            RENORMALIZE=self.renormalize,
            PRESELECT_TOP2=preselect_top2,
            GATHER_TOP2_LOGITS=gather_top2,
            USE_BITONIC_TOP2=bitonic_top2,
            PARALLEL_ROUTES=route_parallel,
            ROUTES_PER_PROGRAM=routes_per_program,
            FIXED_CONTIGUOUS_LAYOUT=fixed_contiguous_layout,
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
