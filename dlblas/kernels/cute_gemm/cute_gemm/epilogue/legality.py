"""Pure host-side legality rules shared by epilogue planning and tuning.

This module intentionally knows nothing about benchmarking, cache identity, or
CUDA launches.  It answers one question only: which GEMM configurations are
legal for this epilogue call.  Sink-buffer capacity is a call-boundary
contract, validated once by the generic tuned wrapper; it must not alter the
candidate set or the cache winner.
"""

from __future__ import annotations

from cute_gemm.cute_dsl_utils import get_device_capacity
from cute_gemm.gemm_config import blockscaled_config_ok, config_supports, cta_tile_shape_m


def _gemm_mn(A, B, b_kn):
    n = B.shape[-1] if b_kn else B.shape[-2]
    m = A.shape[-2] if A.ndim == 3 else A.shape[0]
    return m, n


def sink_config_is_legal(op, *, m, n, tile_m, tile_n, varlen_m):
    """Shared boolean form of the host sink reduction contract."""
    if varlen_m and getattr(op, "dim", 0) == 1 and getattr(op, "combine", "add") != "add":
        return False
    if getattr(op, "check_oob", True) is not False:
        return True
    if getattr(op, "dim", 0) == 0:
        return n % tile_n == 0
    return not varlen_m and m % tile_m == 0


def validate_sink_config(sink_name, op, *, m, n, tile_m, tile_n, varlen_m):
    """Raise the user-facing error for an unsupported sink configuration."""
    if varlen_m and getattr(op, "dim", 0) == 1 and getattr(op, "combine", "add") != "add":
        raise ValueError(
            f"sink '{sink_name}': combine={op.combine!r} M-fold reduces are not "
            "supported under varlen_m"
        )
    if getattr(op, "check_oob", True) is False:
        if getattr(op, "dim", 0) == 0:
            if n % tile_n:
                raise ValueError(
                    f"sink '{sink_name}': check_oob=False requires N divisible by tile_N "
                    f"(N={n}, tile_N={tile_n})"
                )
        elif varlen_m or m % tile_m:
            raise ValueError(
                f"sink '{sink_name}': check_oob=False requires M divisible by the per-CTA tile "
                f"and no varlen_m (M={m}, cta_tile_M={tile_m})"
            )


def prune_epilogue_configs(
    mod, transform_a, configs, named_args, *, device_capacity=None, **kwargs
):
    """Return the structurally legal configs for one epilogue call."""
    kwargs = named_args | kwargs
    A, B = kwargs["A"], kwargs["B"]
    n_full = transform_a.padded_n(B) if transform_a is not None else None
    cap = (
        get_device_capacity(A.device)[0]
        if device_capacity is None
        else (device_capacity[0] if isinstance(device_capacity, tuple) else int(device_capacity))
    )
    A_idx = kwargs.get("A_idx")
    m_gemm, n_gemm = _gemm_mn(A, B, kwargs.get("b_kn", False))
    if A_idx is not None:
        m_gemm = A_idx.shape[0]
    has_out = bool(getattr(mod, "outputs", ()))
    survivors = []
    b_kn_call = kwargs.get("b_kn", False)
    varlen_m = kwargs.get("cu_seqlens_m") is not None
    varlen_or_gather = varlen_m or A_idx is not None
    blockscaled = kwargs.get("SFA") is not None
    has_concat = bool(kwargs.get("concat_layout"))
    for conf in configs:
        c = conf.kwargs["config"]
        if c.device_capacity != cap:
            continue
        if not config_supports(c, gather_A=A_idx is not None, varlen_m=varlen_m):
            continue
        if transform_a is not None:
            if not transform_a.config_ok(c):
                continue
            if n_full is not None and n_full % c.tile_m:
                continue
        if blockscaled and not blockscaled_config_ok(c):
            continue
        if c.swap_ab and (
            not b_kn_call
            or varlen_or_gather
            or has_concat
            or getattr(mod, "mode", None) != "element"
            or mod.sinks
        ):
            continue
        if getattr(mod, "mode", None) == "acc_pair":
            if c.tile_n % 2:
                continue
            if cap == 9 and has_out and c.tile_n % 32:
                continue
        ok = True
        cta_tile_m = cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity, blockscaled)
        for name, op in mod.sinks.items():
            if not sink_config_is_legal(
                op, m=m_gemm, n=n_gemm, tile_m=cta_tile_m, tile_n=c.tile_n, varlen_m=varlen_m
            ):
                ok = False
        if ok:
            survivors.append(conf)
    return survivors


__all__ = ["prune_epilogue_configs", "sink_config_is_legal", "validate_sink_config"]
