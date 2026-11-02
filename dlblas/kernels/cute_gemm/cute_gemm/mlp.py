# Copyright (c) 2025, Tri Dao
from functools import partial
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from cute_gemm.activation import gate_fn_map
from cute_gemm.gemm_base import GemmBase
from cute_gemm.gemm_config import TuneMode
from cute_gemm.gemm_interface import (
    act_to_pytorch_fn_map,
    gated_to_pytorch_fn_map,
    gemm,
    gemm_add_inplace,
)
from cute_gemm.linear import (
    _ops_for,
    _prebuild_running_twins,
    _recompute_act_postact,
    _recompute_gated_postact,
    act_linear_func,
    gated_linear_func,
    linear_act_func,
    linear_fwd_convert_type,
    linear_gated_func,
)

Activation = Literal[
    "gelu_tanh_approx",
    "silu",
    "silu-tanh",
    "relu",
    "relu_sq",
    "swiglu",
    "swiglu-tanh",
    "swiglu_oai",
    "swiglu_oai-tanh",
    "reglu",
    "geglu",
    "glu",
]


# --- Ops bundles for MLP recompute variants ---


class _MLPOps:
    matmul_fwd = gemm
    matmul_fwd_act = GemmBase.gemm_with_activation
    matmul_bwd_dact = partial(GemmBase.gemm_with_activation_grad, dynamic_scheduler=True)
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw_inplace = partial(gemm_add_inplace, dynamic_scheduler=True)
    recompute_postact = staticmethod(_recompute_act_postact)


class _MLPStaticOps:
    matmul_fwd = partial(gemm, tune_mode=TuneMode.STATIC)
    matmul_fwd_act = partial(GemmBase.gemm_with_activation, tune_mode=TuneMode.STATIC)
    matmul_bwd_dact = partial(
        GemmBase.gemm_with_activation_grad, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC)
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )
    recompute_postact = staticmethod(_recompute_act_postact)


class _MLPGatedOps(_MLPOps):
    # Gated math is selected by the activation name inside both class entries.
    matmul_fwd_act = GemmBase.gemm_with_activation
    matmul_bwd_dact = partial(GemmBase.gemm_with_activation_grad, dynamic_scheduler=True)
    recompute_postact = staticmethod(_recompute_gated_postact)


class _MLPGatedStaticOps(_MLPStaticOps):
    matmul_fwd_act = partial(GemmBase.gemm_with_activation, tune_mode=TuneMode.STATIC)
    matmul_bwd_dact = partial(
        GemmBase.gemm_with_activation_grad, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )
    recompute_postact = staticmethod(_recompute_gated_postact)


class _MLPGatedConcatOps(_MLPGatedOps):
    matmul_fwd_act = partial(GemmBase.gemm_with_activation, concat_layout=("B",))
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, concat_layout=("B",))
    matmul_bwd_dw1 = partial(gemm, dynamic_scheduler=True, concat_layout=("out",))
    matmul_bwd_dw1_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, concat_layout=("C", "out")
    )
    recompute_fwd = partial(gemm, concat_layout=("B",))


class _MLPGatedConcatStaticOps(_MLPGatedStaticOps):
    matmul_fwd_act = partial(
        GemmBase.gemm_with_activation, tune_mode=TuneMode.STATIC, concat_layout=("B",)
    )
    matmul_bwd_dx = partial(
        gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC, concat_layout=("B",)
    )
    matmul_bwd_dw1 = partial(
        gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC, concat_layout=("out",)
    )
    matmul_bwd_dw1_inplace = partial(
        gemm_add_inplace,
        dynamic_scheduler=True,
        tune_mode=TuneMode.STATIC,
        concat_layout=("C", "out"),
    )
    recompute_fwd = partial(gemm, tune_mode=TuneMode.STATIC, concat_layout=("B",))


# The RUNNING twins for the bundles _ops_for receives here, built at import
# (same fullgraph-tracing constraint as cute_gemm.linear).
_prebuild_running_twins(_MLPOps, _MLPGatedOps, _MLPGatedConcatOps)


class MLPRecomputeFunc(torch.autograd.Function):
    """MLP with activation recomputation: saves only x (not preact) to reduce memory.

    In backward, recomputes preact = x @ W1.T (one extra matmul) instead of loading it
    from saved tensors. This trades compute for memory:
      - Saves: batch * 2 * hidden * dtype_size bytes of activation memory
      - Costs: one extra GEMM (x @ W1.T) during backward

    Ops class selects between non-gated and gated activation math (chosen by the
    activation name inside gemm_with_activation / gemm_with_activation_grad),
    as well as tuned/untuned.
    """

    @staticmethod
    def forward(ctx, x, weight1, weight2, activation, fuse_grad_accum, ops):
        x, weight1, weight2 = linear_fwd_convert_type(x, weight1, weight2)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight1.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.activation = activation
            ctx.ops = ops
            weight1_og, weight2_og = weight1, weight2
            batch_shape = x.shape[:-1]
            x_flat = x.reshape(-1, x.shape[-1])
            _preact, postact = ops.matmul_fwd_act(x_flat, weight1.T, activation=activation)
            out = ops.matmul_fwd(postact, weight2.T)
            # Save only x and weights — no preact (the whole point of recompute)
            needs_input_grad = ctx.needs_input_grad
            any_grad = needs_input_grad[0] or needs_input_grad[1] or needs_input_grad[2]
            need_dact = needs_input_grad[0] or needs_input_grad[1]  # grad GEMM for dpreact
            saved_x = x if any_grad else None  # recompute preact = x @ W1.T
            saved_w1 = weight1 if any_grad else None  # recompute + dx
            saved_w2 = weight2 if need_dact else None  # only the grad GEMM needs W2
            ctx.save_for_backward(
                saved_x,
                saved_w1,
                saved_w2,
                weight1_og if fuse_grad_accum else None,
                weight2_og if fuse_grad_accum else None,
            )
            return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            x, weight1, weight2, weight1_og, weight2_og = ctx.saved_tensors
            batch_shape = dout.shape[:-1]
            dout = dout.reshape(-1, dout.shape[-1]).contiguous()
            # Recompute preact = x @ W1.T (the extra matmul we trade for memory)
            x_flat = x.reshape(-1, x.shape[-1]) if x is not None else None
            need_dact = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
            any_grad = need_dact or ctx.needs_input_grad[2]
            # concat ops override recompute_fwd to produce interleaved preact matching forward
            recompute_fwd = getattr(ops, "recompute_fwd", ops.matmul_fwd)
            if need_dact:
                preact = recompute_fwd(x_flat, weight1.T)
                # The grad GEMM computes dpreact = d_act(dout @ W2, preact) AND
                # recomputes postact
                dpreact, postact = ops.matmul_bwd_dact(
                    dout, weight2, preact, activation=ctx.activation
                )
            elif any_grad:
                # Only dW2 needed: recompute postact from preact cheaply (no grad GEMM)
                preact = recompute_fwd(x_flat, weight1.T)
                postact = ops.recompute_postact(preact, ctx.activation)
                dpreact = None
            else:
                dpreact, postact = None, None
            # dW2 = dout.T @ postact
            dweight2 = _compute_weight_grad(
                ctx,
                dout,
                postact,
                weight2_og,
                ops.matmul_bwd_dw,
                ops.matmul_bwd_dw_inplace,
                ctx.needs_input_grad[2],
            )
            # dx = dpreact @ W1
            if ctx.needs_input_grad[0]:
                dx = ops.matmul_bwd_dx(dpreact, weight1)
                dx = dx.reshape(*batch_shape, dx.shape[-1])
            else:
                dx = None
            # dW1 = dpreact.T @ x (use dw1 ops if available, e.g. concat layout)
            dw1_fn = getattr(ops, "matmul_bwd_dw1", ops.matmul_bwd_dw)
            dw1_inplace_fn = getattr(ops, "matmul_bwd_dw1_inplace", ops.matmul_bwd_dw_inplace)
            dweight1 = _compute_weight_grad(
                ctx,
                dpreact,
                x_flat,
                weight1_og,
                dw1_fn,
                dw1_inplace_fn,
                ctx.needs_input_grad[1],
            )
            return dx, dweight1, dweight2, None, None, None


def _compute_weight_grad(ctx, dout, x, weight_og, matmul_fn, matmul_inplace_fn, needs_grad):
    if not needs_grad:
        return None
    x = x.reshape(-1, x.shape[-1])
    if not ctx.fuse_grad_accum or weight_og.grad is None or torch.compiler.is_compiling():
        return matmul_fn(dout.T, x, out_dtype=ctx.weight_dtype)
    else:
        matmul_inplace_fn(dout.T, x, weight_og.grad)
        dweight = weight_og.grad
        weight_og.grad = None
        return dweight


def mlp_func(
    x,
    weight1,
    weight2,
    activation: str,
    bias1=None,
    bias2=None,
    fuse_grad_accum=False,
    tune_mode=TuneMode.TUNED,
    recompute=False,
    concat_layout=False,
):
    gated = activation in gate_fn_map
    if concat_layout:
        assert gated, "concat_layout is only supported for gated MLP"
    if recompute:
        if concat_layout:
            ops = _ops_for(tune_mode, _MLPGatedConcatOps, _MLPGatedConcatStaticOps)
        elif gated:
            ops = _ops_for(tune_mode, _MLPGatedOps, _MLPGatedStaticOps)
        else:
            ops = _ops_for(tune_mode, _MLPOps, _MLPStaticOps)
        return MLPRecomputeFunc.apply(x, weight1, weight2, activation, fuse_grad_accum, ops)
    fc1_fn = linear_gated_func if gated else linear_act_func
    fc2_fn = gated_linear_func if gated else act_linear_func
    preact, postact = fc1_fn(
        x,
        weight1,
        activation,
        bias=bias1,
        store_preact=torch.is_grad_enabled(),
        fuse_grad_accum=fuse_grad_accum,
        tune_mode=tune_mode,
        **({"concat_layout": concat_layout} if concat_layout and gated else {}),
    )
    out = fc2_fn(
        preact,
        weight2,
        postact,
        activation=activation,
        bias=bias2,
        fuse_grad_accum=fuse_grad_accum,
        tune_mode=tune_mode,
    )
    return out


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=False,
        bias2=False,
        activation: Activation = "gelu_tanh_approx",
        multiple_of=1,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
        tune_mode: TuneMode = TuneMode.TUNED,
        recompute: bool = False,
        concat_layout: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.activation = activation
        self.gated = activation in gate_fn_map
        assert not concat_layout or self.gated, "concat_layout is only supported for gated MLP"
        if hidden_features is None:
            hidden_features = int(8 / 3 * in_features) if self.gated else 4 * in_features
        if multiple_of > 1:
            hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        fc1_out = 2 * hidden_features if self.gated else hidden_features
        self.fc1 = nn.Linear(in_features, fc1_out, bias=bias1, **factory_kwargs)
        if self.gated:
            if concat_layout:
                self.fc1.weight._muon_reshape_functions = (
                    lambda w: rearrange(w, "(two d) e -> two d e", two=2),
                    lambda w: rearrange(w, "two d e -> (two d) e"),
                )
            else:
                self.fc1.weight._muon_reshape_functions = (
                    lambda w: rearrange(w, "(d two) e -> two d e", two=2),
                    lambda w: rearrange(w, "two d e -> (d two) e"),
                )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.fuse_grad_accum = fuse_grad_accum
        self.tune_mode = tune_mode
        self.recompute = recompute
        self.concat_layout = concat_layout

    def forward(self, input: Tensor) -> Tensor:
        # Allow bias in the fused path during inference (fwd-only, no bwd).
        bias_ok = not torch.is_grad_enabled() or (self.fc1.bias is None and self.fc2.bias is None)
        if (
            bias_ok
            and input.is_cuda
            and input.stride(-1) == 1
            and self.fc1.in_features % 8 == 0
            and self.fc1.out_features % (16 if self.gated else 8) == 0
            and self.fc2.out_features % 8 == 0
        ):
            return mlp_func(
                input,
                self.fc1.weight,
                self.fc2.weight,
                activation=self.activation,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                fuse_grad_accum=self.fuse_grad_accum,
                tune_mode=self.tune_mode,
                recompute=self.recompute,
                concat_layout=self.concat_layout,
            )
        else:
            y = self.fc1(input)
            if self.gated:
                if self.concat_layout:
                    gate, up = y.chunk(2, dim=-1)
                    y = gated_to_pytorch_fn_map[self.activation](gate, up)
                else:
                    y = gated_to_pytorch_fn_map[self.activation](y[..., ::2], y[..., 1::2])
            else:
                y = act_to_pytorch_fn_map[self.activation](y)
            return self.fc2(y)


__all__ = ["MLP"]
