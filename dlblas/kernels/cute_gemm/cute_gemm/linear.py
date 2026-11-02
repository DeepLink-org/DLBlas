# Copyright (c) 2025, Tri Dao
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cute_gemm.gemm_base import GemmBase
from cute_gemm.gemm_config import TuneMode
from cute_gemm.gemm_interface import (
    act_to_pytorch_fn_map,
    gated_to_pytorch_fn_map,
    gemm,
    gemm_add_inplace,
)


def _ensure_contiguous(t):
    """Ensure last-dim stride is 1. Under torch.compile use unconditional .contiguous()
    (dynamo can't inspect strides on fake tensors); otherwise check first to avoid copies.
    """
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


def linear_fwd_convert_type(*tensors):
    autocast_dtype = torch.get_autocast_dtype("cuda")
    if torch.is_autocast_enabled():
        tensors = tuple(t.to(dtype=autocast_dtype) for t in tensors)
    return tensors


def linear_fwd_postprocess(ctx, x, weight, weight_og, needs_x_w_grad):
    needs_input_grad, needs_weight_grad = needs_x_w_grad
    if not needs_input_grad:
        weight, weight_og = None, None
    if not needs_weight_grad:
        x = None
    ctx.save_for_backward(x, weight, weight_og if ctx.fuse_grad_accum else None)


def linear_bwd_compute_input_grad(ctx, dout, weight, matmul_fn):
    if ctx.needs_input_grad[0]:
        assert weight is not None
        return matmul_fn(dout, weight)
    else:
        return None


def linear_bwd_compute_weight_grad(ctx, dout, x, weight_og, matmul_fn, matmul_inplace_fn):
    if ctx.needs_input_grad[1]:
        assert x is not None
        x = x.flatten(0, -2)
        # fuse_grad_accum is not compatible with torch.compile
        if not ctx.fuse_grad_accum or weight_og.grad is None or torch.compiler.is_compiling():
            dweight = matmul_fn(dout.T, x, out_dtype=ctx.weight_dtype)
        else:
            # print("Using fuse grad accum in Linear", dout.shape, x.shape, weight_og.grad.shape)
            matmul_inplace_fn(dout.T, x, weight_og.grad)
            dweight = weight_og.grad
            weight_og.grad = None  # So that pytorch doesn't add dweight to weight_og.grad again
    else:
        dweight = None
    return dweight


def _recompute_act_postact(preact, activation):
    """Recompute postact from preact using the activation function (no GEMM)."""
    return act_to_pytorch_fn_map[activation](preact)


def _recompute_gated_postact(preact, activation):
    """Recompute gated postact from interleaved preact (no GEMM)."""
    return gated_to_pytorch_fn_map[activation](preact[..., ::2], preact[..., 1::2])


# --- Ops bundles: matmul function configurations ---
# Each ops class is a namespace holding the matmul functions for a specific variant
# (default/STATIC, act/gated, etc.). Passed as a non-tensor arg to apply() and stored on ctx.


class _LinearOps:
    matmul_fwd_fn = gemm
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw_inplace = partial(gemm_add_inplace, dynamic_scheduler=True)


class _LinearStaticOps(_LinearOps):
    matmul_fwd_fn = partial(gemm, tune_mode=TuneMode.STATIC)
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC)
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )


class _LinearActOps(_LinearOps):
    matmul_fwd_fn = GemmBase.gemm_with_activation


class _LinearActStaticOps(_LinearStaticOps):
    matmul_fwd_fn = partial(GemmBase.gemm_with_activation, tune_mode=TuneMode.STATIC)


class _LinearGatedOps(_LinearOps):
    # Gated math is selected by the activation name inside gemm_with_activation.
    matmul_fwd_fn = GemmBase.gemm_with_activation


class _LinearGatedStaticOps:
    matmul_fwd_fn = partial(GemmBase.gemm_with_activation, tune_mode=TuneMode.STATIC)
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC)
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )


class _LinearGatedConcatOps(_LinearGatedOps):
    matmul_fwd_fn = partial(GemmBase.gemm_with_activation, concat_layout=("B", "bias"))
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True, concat_layout=("B",))
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True, concat_layout=("out",))
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace, dynamic_scheduler=True, concat_layout=("C", "out")
    )


class _LinearGatedConcatStaticOps(_LinearGatedStaticOps):
    matmul_fwd_fn = partial(
        GemmBase.gemm_with_activation, tune_mode=TuneMode.STATIC, concat_layout=("B", "bias")
    )
    matmul_bwd_dx = partial(
        gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC, concat_layout=("B",)
    )
    matmul_bwd_dw = partial(
        gemm, dynamic_scheduler=True, tune_mode=TuneMode.STATIC, concat_layout=("out",)
    )
    matmul_bwd_dw_inplace = partial(
        gemm_add_inplace,
        dynamic_scheduler=True,
        tune_mode=TuneMode.STATIC,
        concat_layout=("C", "out"),
    )


class _DActLinearOps(_LinearOps):
    matmul_bwd_dx = partial(GemmBase.gemm_with_activation_grad, dynamic_scheduler=True)
    recompute_postact = staticmethod(_recompute_act_postact)


class _DActLinearStaticOps(_LinearStaticOps):
    matmul_bwd_dx = partial(
        GemmBase.gemm_with_activation_grad, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )
    recompute_postact = staticmethod(_recompute_act_postact)


class _DGatedLinearOps(_LinearOps):
    # Gated math is selected by the activation name inside the grad entry.
    matmul_bwd_dx = partial(GemmBase.gemm_with_activation_grad, dynamic_scheduler=True)
    recompute_postact = staticmethod(_recompute_gated_postact)


class _DGatedLinearStaticOps(_LinearStaticOps):
    matmul_bwd_dx = partial(
        GemmBase.gemm_with_activation_grad, dynamic_scheduler=True, tune_mode=TuneMode.STATIC
    )
    recompute_postact = staticmethod(_recompute_gated_postact)


def _mode_twin(base: type, mode: TuneMode) -> type:
    """An ops bundle with ``tune_mode=mode`` baked into every matmul callable.

    Composes along the MRO (a subclass overriding one member keeps the rest);
    existing partials rebake their keywords, bare callables wrap afresh, and
    non-matmul members (recompute helpers) carry through untouched.
    """
    members: dict = {}
    for klass in reversed(base.__mro__):
        for name, fn in vars(klass).items():
            if name.startswith("__") or isinstance(fn, (staticmethod, classmethod)):
                if not name.startswith("__"):
                    members[name] = fn
                continue
            keywords = getattr(fn, "keywords", None)
            if keywords is not None:
                members[name] = partial(fn.func, *fn.args, **{**keywords, "tune_mode": mode})
            elif callable(fn):
                members[name] = partial(fn, tune_mode=mode)
    return type(f"{base.__name__}{mode.name.title()}", (), members)


def _prebuild_running_twins(*bases: type) -> None:
    """Build the RUNNING twins for ``bases`` now, at module import.

    ``_ops_for`` must not allocate on any call path: under
    torch.compile(fullgraph=True) a lazily built twin traces ``_mode_twin``
    (it walks ``vars(klass).items()``), which dynamo cannot trace. Modules
    defining their own ops bundles call this once at module scope.
    """
    for base in bases:
        _RUNNING_TWINS[base] = _mode_twin(base, TuneMode.RUNNING)


# All RUNNING twins _ops_for can receive, built before any tracing can happen.
# mlp.py appends its own bundles at its import.
_RUNNING_TWINS: dict = {}
_prebuild_running_twins(
    _LinearOps,
    _LinearActOps,
    _LinearGatedOps,
    _LinearGatedConcatOps,
    _DActLinearOps,
    _DGatedLinearOps,
)


def _ops_for(tune_mode, base: type, static_ops: type) -> type:
    """Mode -> ops bundle.

    STATIC and RUNNING bake the mode into the callables; every other value
    (the TUNED default, legacy None) rides the bundle's own default -- the
    pre-TuneMode behaviour: a miss is swept. RUNNING twins are pre-built at
    import, so this is a pure selection with nothing left to trace.
    """
    if tune_mode == TuneMode.STATIC:
        return static_ops
    if tune_mode == TuneMode.RUNNING:
        return _RUNNING_TWINS[base]
    return base


# --- Autograd Functions (all @staticmethod, torch.compile-compatible) ---


class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, fuse_grad_accum, ops):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
        out: (..., out_features)
        """
        # Convert types while autocast is still enabled, then disable it for the body.
        x, weight = linear_fwd_convert_type(x, weight)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.ops = ops
            weight_og = weight
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            out = ops.matmul_fwd_fn(x, weight.T, bias=bias)
            linear_fwd_postprocess(
                ctx, x, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2]
            )
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.compute_dbias = bias is not None and ctx.needs_input_grad[2]
            return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            x, weight, weight_og = ctx.saved_tensors  # weight_og is None if not ctx.fuse_grad_accum
            batch_shape = dout.shape[:-1]
            dout = _ensure_contiguous(dout.flatten(0, -2))
            dbias = dout.sum(0, dtype=ctx.bias_dtype) if ctx.compute_dbias else None
            dx = linear_bwd_compute_input_grad(ctx, dout, weight, ops.matmul_bwd_dx)
            dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
            dweight = linear_bwd_compute_weight_grad(
                ctx, dout, x, weight_og, ops.matmul_bwd_dw, ops.matmul_bwd_dw_inplace
            )
            return dx, dweight, dbias, None, None


def linear_func(x, weight, bias=None, fuse_grad_accum=False, tune_mode=TuneMode.TUNED):
    ops = _ops_for(tune_mode, _LinearOps, _LinearStaticOps)
    return LinearFunc.apply(x, weight, bias, fuse_grad_accum, ops)


class LinearActFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation, bias, store_preact, fuse_grad_accum, ops):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
        out: (..., out_features)
        Return both out and post-activation, but only out is differentiable.
        """
        x, weight = linear_fwd_convert_type(x, weight)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.ops = ops
            weight_og = weight
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            out, postact = ops.matmul_fwd_fn(
                x, weight.T, bias=bias, activation=activation, store_preact=store_preact
            )
            linear_fwd_postprocess(
                ctx, x, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2]
            )
            if out is not None:
                out = out.reshape(*batch_shape, out.shape[-1])
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.compute_dbias = bias is not None and ctx.needs_input_grad[3]
            ctx.mark_non_differentiable(postact)
            ctx.set_materialize_grads(False)  # We don't want to materialize grads for postact
            return out, postact.reshape(*batch_shape, postact.shape[-1])

    @staticmethod
    def backward(ctx, dout, *args):
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            x, weight, weight_og = ctx.saved_tensors
            batch_shape = dout.shape[:-1]
            dout = _ensure_contiguous(dout.flatten(0, -2))
            dbias = dout.sum(0, dtype=ctx.bias_dtype) if ctx.compute_dbias else None
            dx = linear_bwd_compute_input_grad(ctx, dout, weight, ops.matmul_bwd_dx)
            dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
            dweight = linear_bwd_compute_weight_grad(
                ctx, dout, x, weight_og, ops.matmul_bwd_dw, ops.matmul_bwd_dw_inplace
            )
            return dx, dweight, None, dbias, None, None, None


def linear_act_func(
    x,
    weight,
    activation,
    bias=None,
    store_preact=True,
    fuse_grad_accum=False,
    tune_mode=TuneMode.TUNED,
):
    ops = _ops_for(tune_mode, _LinearActOps, _LinearActStaticOps)
    return LinearActFunc.apply(x, weight, activation, bias, store_preact, fuse_grad_accum, ops)


def linear_gated_func(
    x,
    weight,
    activation,
    bias=None,
    store_preact=True,
    fuse_grad_accum=False,
    tune_mode=TuneMode.TUNED,
    concat_layout=False,
):
    if concat_layout:
        ops = _ops_for(tune_mode, _LinearGatedConcatOps, _LinearGatedConcatStaticOps)
    else:
        ops = _ops_for(tune_mode, _LinearGatedOps, _LinearGatedStaticOps)
    return LinearActFunc.apply(x, weight, activation, bias, store_preact, fuse_grad_accum, ops)


class DActLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, preact, weight, x, activation, bias, fuse_grad_accum, ops):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        bias: (out_features,) or None
        out: (..., out_features)
        Takes in an extra preact argument which is the pre-activation, to be used in the backward pass.
        """
        x, weight = linear_fwd_convert_type(x, weight)
        with torch.amp.autocast("cuda", enabled=False):
            ctx.weight_dtype = weight.dtype
            ctx.fuse_grad_accum = fuse_grad_accum
            ctx.ops = ops
            weight_og = weight
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            out = ops.matmul_fwd_fn(x, weight.T, bias=bias)
            # Store preact instead of x, we will recompute x (postact) in backward.
            # dpreact needs gemm_with_activation_grad(dout, weight, preact) → needs both
            # weight and preact.
            # dweight needs postact: if dpreact is also needed, postact comes from
            # gemm_with_activation_grad;
            # otherwise we can recompute postact = act(preact) cheaply without weight.
            need_preact = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
            need_weight = ctx.needs_input_grad[0]  # only the grad GEMM needs weight
            linear_fwd_postprocess(
                ctx, preact, weight, weight_og, needs_x_w_grad=(need_weight, need_preact)
            )
            ctx.activation = activation
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.compute_dbias = bias is not None and ctx.needs_input_grad[4]
            return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        with torch.amp.autocast("cuda", enabled=False):
            ops = ctx.ops
            # weight_og is None if not ctx.fuse_grad_accum
            preact, weight, weight_og = ctx.saved_tensors
            batch_shape = dout.shape[:-1]
            dout = _ensure_contiguous(dout.flatten(0, -2))
            dbias = dout.sum(0, dtype=ctx.bias_dtype) if ctx.compute_dbias else None
            if ctx.needs_input_grad[0]:
                # Need dpreact: gemm_with_activation_grad(dout, weight, preact)
                # → (dpreact, postact)
                preact = preact.flatten(0, -2)
                assert weight is not None
                dpreact, x = ops.matmul_bwd_dx(dout, weight, preact, activation=ctx.activation)
            elif ctx.needs_input_grad[1]:
                # Only need dweight: recompute postact from preact cheaply (no GEMM needed)
                preact = preact.flatten(0, -2)
                x = ops.recompute_postact(preact, ctx.activation)
                dpreact = None
            else:
                dpreact, x = None, None
            dpreact = (
                dpreact.reshape(*batch_shape, dpreact.shape[-1]) if dpreact is not None else None
            )
            dweight = linear_bwd_compute_weight_grad(
                ctx, dout, x, weight_og, ops.matmul_bwd_dw, ops.matmul_bwd_dw_inplace
            )
            return dpreact, dweight, None, None, dbias, None, None


def act_linear_func(
    preact, weight, x, activation, bias=None, fuse_grad_accum=False, tune_mode=TuneMode.TUNED
):
    ops = _ops_for(tune_mode, _DActLinearOps, _DActLinearStaticOps)
    return DActLinearFunc.apply(preact, weight, x, activation, bias, fuse_grad_accum, ops)


def gated_linear_func(
    preact, weight, x, activation, bias=None, fuse_grad_accum=False, tune_mode=TuneMode.TUNED
):
    ops = _ops_for(tune_mode, _DGatedLinearOps, _DGatedLinearStaticOps)
    return DActLinearFunc.apply(preact, weight, x, activation, bias, fuse_grad_accum, ops)


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fuse_grad_accum = fuse_grad_accum

    def forward(self, input: Tensor) -> Tensor:
        if input.is_cuda and self.in_features % 8 == 0 and self.out_features % 8 == 0:
            return linear_func(input, self.weight, self.bias, fuse_grad_accum=self.fuse_grad_accum)
        else:
            return F.linear(input, self.weight, self.bias)


__all__ = ["linear_act_func", "linear_func", "linear_gated_func"]
