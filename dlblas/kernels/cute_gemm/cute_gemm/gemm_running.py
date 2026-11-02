"""Describe every reachable token bucket for one SM90 GEMM contract.

The result contains only metadata tensors. Compilation and publication belong
to Autotuner; bucket boundaries and tensor roles stay in the GEMM policies.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import replace
from itertools import pairwise

import torch

from cute_gemm.autotuner import AutotuneConfig, RunningConfig
from cute_gemm.cute_dsl_utils import get_device_capacity
from cute_gemm.gemm_heuristic import desc_from_call, rank
from cute_gemm.varlen_selector import (
    _AVERAGE_ROWS_BUCKETS,
    K_GROUP_TREE,
    M_GROUP_TREE,
    TRAINED_ENVELOPE,
)


def _meta_tensor(value, axis=None, rows=None):
    """Keep layout, including padding/broadcasts, while resizing a routed axis."""
    shape, strides = list(value.shape), list(value.stride())
    if axis is not None:
        axis %= value.ndim
        old_rows = shape[axis]
        # Strides above the resized physical dimension scale with its extent.
        # Ratios preserve padding. Strides 0/1 retain their layout meaning.
        span = max(old_rows, 1) * strides[axis]
        if span:
            for dim, stride in enumerate(strides):
                if dim != axis and stride >= span and stride % span == 0:
                    strides[dim] = stride // max(old_rows, 1) * rows
        shape[axis] = rows
    return torch.empty_strided(shape, strides, dtype=value.dtype, device="meta")


def running_configs(
    policy,
    args,
    kwargs,
    configs,
    *,
    mod=None,
    b_kn_default=True,
):
    """One analytical choice per W over the finite average-rows bucket set.

    Split at bucket, selector-leaf and selector-envelope boundaries, then
    deduplicate using the policy's real key. Upper endpoints are canonical
    representatives; no input-sized GPU allocation or prefix-value read occurs.
    Unsupported architectures and packed scale/transform layouts retain their
    existing mode contract rather than borrowing an incompatible ranker.
    """
    named = {
        name: default
        for name, default in zip(policy.arg_names, policy.arg_defaults)
        if default is not inspect.Parameter.empty
    }
    named.update(zip(policy.arg_names, args))
    named.update(kwargs)
    cu_m, cu_k = named.get("cu_seqlens_m"), named.get("cu_seqlens_k")
    if cu_m is None and cu_k is None:
        return None
    A, B = named["A"], named["B"]
    device_capability = get_device_capacity(A.device)
    if device_capability[0] != 9:
        return None
    if any(named.get(n) is not None for n in ("SFA", "SFB", "SFD", "SFDCol")):
        return None
    experts = (cu_m if cu_m is not None else cu_k).shape[0] - 1
    if experts <= 0 or (cu_m is not None and cu_k is not None):
        return None

    starts = {1}
    starts.update(bucket * experts + 1 for bucket in _AVERAGE_ROWS_BUCKETS[:-1])
    tree = M_GROUP_TREE if cu_m is not None else K_GROUP_TREE
    multiplier = experts if cu_m is not None else B.shape[-1]
    for leaf in tree:
        if math.isfinite(leaf["upper"]):
            starts.add(math.floor(leaf["upper"] * multiplier) + 1)
    if cu_k is not None:
        lo, hi = TRAINED_ENVELOPE["K-group"]["ragged_K"]
        starts.update((lo, hi + 1))
    starts = sorted(s for s in starts if s >= 1)

    contract = policy.make_contract(args, kwargs, lambda: None)
    by_key = {}
    tail_start = _AVERAGE_ROWS_BUCKETS[-2] * experts + 1
    representatives = [start if start >= tail_start else end - 1 for start, end in pairwise(starts)]
    representatives.append(starts[-1])  # the open tail has no further identity boundaries
    routed = named.get("A_idx")
    representatives.append(
        routed.shape[0] if routed is not None else A.shape[-2] if cu_m is not None else B.shape[0]
    )
    for rows in representatives:
        call = {}
        for name, value in named.items():
            if not isinstance(value, torch.Tensor):
                call[name] = value
                continue
            axis = None
            if cu_m is not None:
                if name in policy._ragged_m_out_names or (
                    name == "A" and named.get("A_idx") is None
                ):
                    axis = -2
                elif name in policy._routed_m_tensor_names or name == "A_idx":
                    axis = -1
            else:
                if name == "A" and named.get("A_idx") is None:
                    axis = -1
                elif name == "B":
                    axis = 0
                elif name == "A_idx":
                    axis = -1
            call[name] = _meta_tensor(value, axis, rows)
        call.pop("config", None)
        if (
            policy._gemm_key((), call, contract=True, device_capability=device_capability)
            != contract
        ):
            continue
        key = policy._gemm_key((), call, device_capability=device_capability)
        prepared = policy.prepare_candidates(configs, call, {})
        if not prepared:
            raise RuntimeError("RUNNING bucket has no structurally legal GEMM config")
        candidates = policy.shortlist(list(prepared), call, {})
        logical_b = call["B"] if call.get("b_kn", b_kn_default) else call["B"].mT
        desc = desc_from_call(
            call["A"],
            logical_b,
            call.get("out", call.get("D")),
            call.get("cu_seqlens_m"),
            call.get("cu_seqlens_k"),
        )
        if call.get("A_idx") is not None:
            desc = replace(desc, **({"m": rows} if cu_m is not None else {"k": rows}))
        config = rank(desc, [c.kwargs["config"] for c in prepared])
        if config is None:
            raise RuntimeError("RUNNING bucket is not covered by the GEMM analytical ranker")
        # The explicit kernel call requires config-shaped reduction buffers.
        if mod is not None and mod.sinks:
            from cute_gemm.gemm_runtime.autotune import sink_arg_shapes

            # Reuse the runtime's worst-case capacity: a disk winner may
            # replace the analytical config before capture and slice it.
            for name, shape in sink_arg_shapes(
                mod, rows, desc.n, device="meta", num_seqs=experts
            ).items():
                call[name] = torch.empty(shape, dtype=torch.float32, device="meta")
        by_key[key] = RunningConfig(key, call, AutotuneConfig(config=config), candidates)
    return list(by_key.values())
