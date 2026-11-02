# Copyright (c) 2026, cute_gemm team.
# Derived from cute_gemm.gemm_sm90.py; repository license remains Apache-2.0.

"""SM90 MoE role class: the MoE host call contract plus an inherited device core.

``GemmMoeSm90`` owns everything MoE-specific on the host side: the activation
and W13 layout whitelists, the public tensor-contract validation, and the two
execution entry points (:meth:`up_proj_act`, :meth:`down_proj_dact`) that map
MoE operands onto the class-owned ``gemm_with_activation`` /
``gemm_with_activation_grad`` entries. The device side inherits the plain
16-bit mainloop from :class:`cute_gemm.gemm_sm90.GemmSm90`
unchanged — the constructor's guard (16-bit A, no RS mainloop, no operand
transform) already pins exactly the path the generic class specializes for
that case. The subclass is kept for the role's independent identity (the
kernel-variant token and its separate semantic digests) and as the override
point for future MoE-specific load, schedule, or MMA changes.

The host methods are plain-Python classmethods: they keep no state on the
class (no Tensors, plans, or outputs) and call the class entries inherited
from ``GemmBase`` — the entries take the ``kernel_variant`` token from the
class themselves — so this role module holds no dependency on the generic
GEMM interface module at all.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cute_gemm.gemm_config import SplitKMode, TuneMode
from cute_gemm.gemm_sm90 import GemmSm90


class GemmMoeSm90(GemmSm90):
    """SM90 MoE GEMM role: host call contract, restricted construction, inherited mainloop.

    The class constants and the two host classmethods define the complete MoE
    call contract — whitelists, input validation, operand mapping, and the
    strict return unpack. The constructor narrows the generic device contract
    to plain BF16/FP16 WGMMA; every device-side stage, the MMA loop included,
    comes from ``GemmSm90`` (which today schedules MoE work with the same tile
    scheduler as every other SM90 GEMM). MoE-specific device changes belong
    here as method overrides when they arrive.
    """

    # The class-owned wire token (see GemmBase.variant_id): this string, not
    # a module:qualname, crosses the kernel_variant custom-op field, so the
    # compiled graph survives refactors of this module.
    variant_id = "moe_sm90"
    _ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)
    _ROUTER_DTYPES = (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    _GATED_ACTIVATIONS = ("swiglu", "geglu", "reglu")
    _ACTIVATIONS = _GATED_ACTIVATIONS + ("silu", "relu", "relu_sq", "gelu_tanh_approx")
    _GATE_UP_LAYOUTS = ("concat", "interleaved")

    def __init__(
        self,
        acc_dtype,
        a_dtype,
        tile_shape_mnk: tuple[int, int] | tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        fp8_fast_accum: bool = False,
        gather_A: bool = False,
        use_clc_persistence: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
        split_k: int = 1,
        split_k_mode: int = SplitKMode.SERIAL,
        mma_is_rs: bool = False,
        transform_a=None,
    ):
        if a_dtype.width != 16 or mma_is_rs or transform_a is not None:
            raise ValueError(
                "GemmMoeSm90 supports only plain BF16/FP16 WGMMA; "
                "RS and operand-transform mainloops are not part of the MoE contract"
            )
        super().__init__(
            acc_dtype,
            a_dtype,
            tile_shape_mnk,
            cluster_shape_mnk,
            pingpong=pingpong,
            is_persistent=is_persistent,
            fp8_fast_accum=fp8_fast_accum,
            gather_A=gather_A,
            use_clc_persistence=use_clc_persistence,
            concat_layout=concat_layout,
            use_pdl=use_pdl,
            split_k=split_k,
            split_k_mode=split_k_mode,
            mma_is_rs=mma_is_rs,
            transform_a=transform_a,
        )

    # ------------------------------------------------------------------
    # Host execution: the internal MoE entry points
    # ------------------------------------------------------------------

    @classmethod
    def up_proj_act(
        cls,
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
        """Validate and run the expert up projection (see ``moe_up_proj_act``)."""
        cls._validate_up_inputs(
            x,
            gate_up_weight,
            expert_offsets,
            token_indices,
            gate_up_bias,
            activation,
            gate_up_layout,
        )
        gated = activation in cls._GATED_ACTIVATIONS
        preact, postact = cls.gemm_with_activation(
            x,
            gate_up_weight.mT,
            bias=gate_up_bias,
            activation=activation,
            cu_seqlens_m=expert_offsets,
            A_idx=token_indices,
            store_preact=store_preact,
            dynamic_scheduler=True,
            tune_mode=tune_mode,
            concat_layout=(
                (("B", "bias") if gate_up_bias is not None else ("B",))
                if gated and gate_up_layout == "concat"
                else None
            ),
        )
        return postact, preact

    @classmethod
    def down_proj_dact(
        cls,
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
        """Validate and run the routed down projection (see ``moe_down_proj_dact``)."""
        cls._validate_down_inputs(
            grad_output,
            down_weight,
            gate_up_preact,
            expert_offsets,
            token_indices,
            router_weights,
            compute_router_weight_grad,
            activation,
        )
        result = cls.gemm_with_activation_grad(
            grad_output,
            down_weight,
            gate_up_preact,
            activation=activation,
            colvec_scale=router_weights,
            colvec_reduce=compute_router_weight_grad,
            cu_seqlens_m=expert_offsets,
            A_idx=token_indices,
            dynamic_scheduler=True,
            tune_mode=tune_mode,
        )
        # gemm_with_activation_grad returns exactly (dx, postact, reduce) with
        # the reduce and exactly (dx, postact) without (the K=0 fast path
        # included). Unpack strictly per the flag: a wrong arity is a contract
        # violation to surface, not something to adapt away by None-filling.
        if compute_router_weight_grad:
            if len(result) != 3:
                raise RuntimeError(
                    f"gemm_with_activation_grad returned {len(result)} value(s); "
                    "expected exactly (dx, postact, router_weight_grad) when "
                    "compute_router_weight_grad=True"
                )
            if result[2] is None:
                raise RuntimeError(
                    "gemm_with_activation_grad returned a None router-weight grad; "
                    "expected a tensor when compute_router_weight_grad=True"
                )
            grad_gate_up, weighted_postact, grad_router_weights = result
        else:
            if len(result) != 2:
                raise RuntimeError(
                    f"gemm_with_activation_grad returned {len(result)} value(s); "
                    "expected exactly (dx, postact) when "
                    "compute_router_weight_grad=False"
                )
            grad_gate_up, weighted_postact = result
            grad_router_weights = None
        return grad_gate_up, weighted_postact, grad_router_weights

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_up_inputs(
        cls,
        x: Tensor,
        gate_up_weight: Tensor,
        expert_offsets: Tensor,
        token_indices: Tensor | None,
        gate_up_bias: Tensor | None,
        activation: str,
        gate_up_layout: str,
    ) -> None:
        cls._check_activation(activation)
        if gate_up_layout not in cls._GATE_UP_LAYOUTS:
            raise ValueError(
                f"gate_up_layout must be one of {cls._GATE_UP_LAYOUTS}, got {gate_up_layout!r}"
            )
        gated = activation in cls._GATED_ACTIVATIONS
        if not gated and gate_up_layout != "concat":
            raise ValueError("gate_up_layout='interleaved' only applies to gated activations")
        if x.ndim != 2:
            raise ValueError(f"x must be [T, D], got {tuple(x.shape)}")
        weight_rows = "2I" if gated else "I"
        if gate_up_weight.ndim != 3:
            raise ValueError(
                f"gate_up_weight must be [E, {weight_rows}, D], got {tuple(gate_up_weight.shape)}"
            )
        if x.dtype not in cls._ACTIVATION_DTYPES:
            raise TypeError(f"x must be float16 or bfloat16, got {x.dtype}")
        if gate_up_weight.dtype != x.dtype:
            raise TypeError("x and gate_up_weight must use the same activation dtype")
        if gate_up_weight.shape[2] != x.shape[1] or (gated and gate_up_weight.shape[1] % 2):
            raise ValueError(
                f"gate_up_weight must have shape [E, {weight_rows}, D] with D=x.shape[1]"
                f"{' and an even 2I dimension' if gated else ''}, "
                f"got x={tuple(x.shape)}, weight={tuple(gate_up_weight.shape)}"
            )
        if x.shape[1] == 0:
            raise ValueError(
                "moe_up_proj_act: zero contraction dim (D=0) is unsupported; got "
                f"x={tuple(x.shape)}, gate_up_weight={tuple(gate_up_weight.shape)}"
            )
        cls._check_same_device(x, gate_up_weight, expert_offsets, token_indices, gate_up_bias)
        cls._check_sm90_if_cuda(x)
        route_count = x.shape[0] if token_indices is None else token_indices.shape[0]
        cls._check_offsets(expert_offsets, gate_up_weight.shape[0], route_count)
        cls._check_indices(token_indices, x.device, route_count, x.shape[0])
        if gate_up_bias is not None:
            if gate_up_bias.shape != (gate_up_weight.shape[0], gate_up_weight.shape[1]):
                raise ValueError(
                    f"gate_up_bias must be [E, {weight_rows}] matching gate_up_weight rows "
                    f"(same {gate_up_layout} layout), got {tuple(gate_up_bias.shape)}"
                )
            if gate_up_bias.dtype not in (*cls._ACTIVATION_DTYPES, torch.float32):
                raise TypeError("gate_up_bias must use the activation dtype or float32")

    @classmethod
    def _validate_down_inputs(
        cls,
        grad_output: Tensor,
        down_weight: Tensor,
        gate_up_preact: Tensor,
        expert_offsets: Tensor,
        token_indices: Tensor | None,
        router_weights: Tensor | None,
        compute_router_weight_grad: bool,
        activation: str,
    ) -> None:
        cls._check_activation(activation)
        gated = activation in cls._GATED_ACTIVATIONS
        preact_rows = "2I" if gated else "I"
        if grad_output.ndim != 2:
            raise ValueError(f"grad_output must be [T, D], got {tuple(grad_output.shape)}")
        if down_weight.ndim != 3:
            raise ValueError(f"down_weight must be [E, D, I], got {tuple(down_weight.shape)}")
        if gate_up_preact.ndim != 2:
            raise ValueError(
                f"gate_up_preact must be [R, {preact_rows}], got {tuple(gate_up_preact.shape)}"
            )
        if grad_output.dtype not in cls._ACTIVATION_DTYPES:
            raise TypeError(f"grad_output must be float16 or bfloat16, got {grad_output.dtype}")
        if down_weight.dtype != grad_output.dtype or gate_up_preact.dtype != grad_output.dtype:
            raise TypeError(
                "grad_output, down_weight, and gate_up_preact must use one activation dtype"
            )
        preact_width_matches = (
            down_weight.shape[2] * 2 == gate_up_preact.shape[1]
            if gated
            else down_weight.shape[2] == gate_up_preact.shape[1]
        )
        if down_weight.shape[1] != grad_output.shape[1] or not preact_width_matches:
            raise ValueError(
                f"expected down_weight=[E, D, I], grad_output=[T, D], "
                f"gate_up_preact=[R, {preact_rows}]{' in interleaved gate/up order' if gated else ''}, "
                f"got grad_output={tuple(grad_output.shape)}, down_weight={tuple(down_weight.shape)}, "
                f"gate_up_preact={tuple(gate_up_preact.shape)}"
            )
        if grad_output.shape[1] == 0:
            raise ValueError(
                "moe_down_proj_dact: zero contraction dim (D=0) is unsupported; got "
                f"grad_output={tuple(grad_output.shape)}, down_weight={tuple(down_weight.shape)}"
            )
        cls._check_same_device(
            grad_output, down_weight, gate_up_preact, expert_offsets, token_indices, router_weights
        )
        cls._check_sm90_if_cuda(grad_output)
        route_count = gate_up_preact.shape[0]
        if token_indices is None and grad_output.shape[0] != route_count:
            raise ValueError(
                "when token_indices is None, grad_output rows must equal gate_up_preact rows "
                f"({route_count}), got {grad_output.shape[0]}"
            )
        cls._check_offsets(expert_offsets, down_weight.shape[0], route_count)
        cls._check_indices(
            token_indices, grad_output.device, gate_up_preact.shape[0], grad_output.shape[0]
        )
        if router_weights is not None:
            if router_weights.ndim != 1 or router_weights.shape[0] != gate_up_preact.shape[0]:
                raise ValueError(
                    f"router_weights must be [R] with R={gate_up_preact.shape[0]}, "
                    f"got {tuple(router_weights.shape)}"
                )
            if router_weights.dtype not in cls._ROUTER_DTYPES:
                raise TypeError(f"router_weights must use one of {cls._ROUTER_DTYPES}")
        if compute_router_weight_grad and router_weights is None:
            raise ValueError("router_weights is required when compute_router_weight_grad=True")

    # ------------------------------------------------------------------
    # Shared checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_same_device(*tensors: Tensor | None) -> None:
        present = [t for t in tensors if t is not None]
        if present and any(t.device != present[0].device for t in present[1:]):
            raise ValueError("all MoE tensors must be on the same device")

    @staticmethod
    def _check_sm90_if_cuda(tensor: Tensor) -> None:
        # CPU inputs remain admissible for host-side API and spy tests. The generic
        # GEMM call rejects CPU execution; CUDA calls get an early architecture
        # error with a useful message.
        if tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] != 9:
            raise RuntimeError("SM90 MoE primitives require an SM90/Hopper device")

    @classmethod
    def _check_activation(cls, activation: str) -> None:
        if activation not in cls._ACTIVATIONS:
            raise ValueError(f"activation must be one of {cls._ACTIVATIONS}, got {activation!r}")

    @staticmethod
    def _check_offsets(
        expert_offsets: Tensor, expert_count: int, route_count: int | None = None
    ) -> None:
        if expert_offsets.ndim != 1 or expert_offsets.shape[0] != expert_count + 1:
            raise ValueError(
                f"expert_offsets must have shape [{expert_count + 1}], got "
                f"{tuple(expert_offsets.shape)}"
            )
        if expert_offsets.dtype != torch.int32:
            raise TypeError(f"expert_offsets must be int32, got {expert_offsets.dtype}")
        if (
            expert_offsets.device.type == "cpu"
            and expert_offsets.numel()
            and int(expert_offsets[0]) != 0
        ):
            raise ValueError(f"expert_offsets[0] must be 0, got {int(expert_offsets[0])}")
        if (
            expert_offsets.device.type == "cpu"
            and expert_offsets.numel() > 1
            and bool((expert_offsets[1:] < expert_offsets[:-1]).any())
        ):
            raise ValueError("expert_offsets must be monotonically nondecreasing")
        if (
            route_count is not None
            and expert_offsets.device.type == "cpu"
            and expert_offsets.numel()
            and int(expert_offsets[-1]) != route_count
        ):
            raise ValueError(
                f"expert_offsets[-1] must equal routed row count {route_count}, "
                f"got {int(expert_offsets[-1])}"
            )

    @staticmethod
    def _check_indices(
        token_indices: Tensor | None,
        device: torch.device,
        route_count: int | None,
        row_count: int | None = None,
    ) -> None:
        if token_indices is None:
            return
        if token_indices.ndim != 1:
            raise ValueError(f"token_indices must be 1-D, got {tuple(token_indices.shape)}")
        if token_indices.dtype != torch.int32:
            raise TypeError(f"token_indices must be int32, got {token_indices.dtype}")
        if token_indices.device != device:
            raise ValueError("token_indices must be on the same device as the activation")
        if route_count is not None and token_indices.shape[0] != route_count:
            raise ValueError(
                f"token_indices length must equal routed row count {route_count}, "
                f"got {token_indices.shape[0]}"
            )
        if row_count is not None and token_indices.device.type == "cpu" and token_indices.numel():
            lo, hi = int(token_indices.min()), int(token_indices.max())
            if lo < 0 or hi >= row_count:
                raise ValueError(
                    f"token_indices values must be in [0, {row_count}), got range [{lo}, {hi}]"
                )


# Explicit static-role registration into the neutral identity registry
# (GemmBase.register_kernel_variant): both execution paths resolve the
# "moe_sm90" token through it. Kept next to the class — never automatic
# (no __init_subclass__), so minted kernel classes stay out.
GemmMoeSm90.register_kernel_variant()

__all__ = ["GemmMoeSm90"]
