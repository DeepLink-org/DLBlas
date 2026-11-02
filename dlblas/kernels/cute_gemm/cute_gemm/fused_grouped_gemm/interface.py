# Copyright (c) 2026, cute_gemm team.

"""The two public SM90 MoE fused grouped-GEMM primitives.

Each op is one expert-grouped GEMM with the activation fused into the epilogue.
The activation whitelist and the W13 storage layouts match sonicmoe's expert
compute surface. Routing, aggregation, weight gradients, and autograd ownership
remain with the caller. The complete MoE host call contract — constants,
validation, GEMM parameter mapping, and the strict return unpack — lives in
:class:`cute_gemm.fused_grouped_gemm.gemm_sm90.GemmMoeSm90`; these entry points
forward to it and add nothing.
"""

from __future__ import annotations

from torch import Tensor

from cute_gemm.gemm_config import TuneMode

from .gemm_sm90 import GemmMoeSm90


def moe_up_proj_act(
    x: Tensor,
    gate_up_weight: Tensor,
    expert_offsets: Tensor,
    *,
    activation: str = "swiglu",
    gate_up_layout: str = "concat",
    token_indices: Tensor | None = None,
    gate_up_bias: Tensor | None = None,
    store_preact: bool = True,
    tune_mode: TuneMode = TuneMode.TUNED,
) -> tuple[Tensor, Tensor | None]:
    """Run the expert up projection with the activation fused in the epilogue.

    ``token_indices[r]`` gathers the original activation row for routed row
    ``r``; with ``token_indices=None`` the activation is already packed in
    expert-major order. ``activation`` selects one of the seven sonicmoe
    activations. Gated activations (swiglu, geglu, reglu) read ``gate_up_weight``
    as ``[E, 2I, D]`` — stored concat ``[all gate; all up]`` by default, or
    interleaved ``[gate0, up0, gate1, ...]`` with ``gate_up_layout=
    "interleaved"``; ``gate_up_bias`` follows the same layout. Ungated
    activations consume the plain ``[E, I, D]`` projection. Returns
    ``(postact, preact)`` in routed row order; ``preact`` is ``None`` when
    ``store_preact=False``. No down projection or aggregation is performed.
    """

    return GemmMoeSm90.up_proj_act(
        x,
        gate_up_weight,
        expert_offsets,
        activation=activation,
        gate_up_layout=gate_up_layout,
        token_indices=token_indices,
        gate_up_bias=gate_up_bias,
        store_preact=store_preact,
        tune_mode=tune_mode,
    )


def moe_down_proj_dact(
    grad_output: Tensor,
    down_weight: Tensor,
    gate_up_preact: Tensor,
    expert_offsets: Tensor,
    *,
    activation: str = "swiglu",
    token_indices: Tensor | None = None,
    router_weights: Tensor | None = None,
    compute_router_weight_grad: bool = False,
    tune_mode: TuneMode = TuneMode.TUNED,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Compute the routed activation input gradient and weighted postactivation.

    The down GEMM accumulator is unscaled. ``router_weights`` scales the dact
    input gradient and returned postactivation. The optional reduction returns
    the activation-path score gradient contribution per routed row. For gated
    activations ``gate_up_preact`` is the interleaved ``[R, 2I]`` preactivation
    produced by :func:`moe_up_proj_act`; for ungated activations it is the
    natural-order ``[R, I]`` preactivation.

    ``router_weights`` supports float32, float16, bfloat16, float8_e4m3fn and
    float8_e5m2. Low-precision scores require stride=1 and a 16-byte-aligned
    base pointer; invalid layouts raise ValueError before tuning or launch.
    Scores retain their storage dtype and convert to float32 inside the
    epilogue. The optional score gradient is float32.
    """

    return GemmMoeSm90.down_proj_dact(
        grad_output,
        down_weight,
        gate_up_preact,
        expert_offsets,
        activation=activation,
        token_indices=token_indices,
        router_weights=router_weights,
        compute_router_weight_grad=compute_router_weight_grad,
        tune_mode=tune_mode,
    )


__all__ = ["moe_down_proj_dact", "moe_up_proj_act"]
