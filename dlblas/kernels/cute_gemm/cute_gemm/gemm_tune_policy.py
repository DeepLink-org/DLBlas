"""Generic GEMM autotune policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from cute_gemm.autotuner import AutotuneConfig, TunePolicy
from cute_gemm.cute_dsl_utils import get_device_capacity
from cute_gemm.gemm_config import (
    GemmConfig,
    blockscaled_config_ok,
    config_supports,
    cta_tile_shape_m,
)


_SPLIT_K_FACTORS = (2, 4, 8, 16)


def _split_k_of(conf):
    """Read the split factor from a typed config (tests may use sentinels)."""
    return getattr(conf.kwargs.get("config"), "split_k", 1)


def _is_separate_split_k(mode) -> bool:
    """Avoid importing the enum through a higher-level GEMM module."""
    try:
        return int(mode) == 2  # SplitKMode.SEPARATE
    except (TypeError, ValueError):
        return str(mode).upper().endswith("SEPARATE")


def _expand_split_k_configs(configs, A):
    """Add every structurally supported split-K factor.

    This is intentionally independent of workload size and SM occupancy.  A
    policy's preparation phase describes the legal search space; performance
    heuristics belong to :meth:`GemmTunePolicy.shortlist` below.
    """
    expanded = list(configs)
    for conf in configs:
        c = conf.kwargs["config"]
        if _split_k_of(conf) != 1:
            continue
        k_tiles = -(-int(A.shape[-1]) // (c.tile_k or 64))
        expanded.extend(
            AutotuneConfig(config=replace(c, split_k=split_k))
            for split_k in _SPLIT_K_FACTORS
            if 2 * split_k <= k_tiles
        )
    return expanded


def _shortlist_split_k_configs(configs, A, B, device_capacity):
    """Apply occupancy/K-tile heuristics to already-legal split-K candidates."""
    try:
        if A.ndim == 3:
            L, M, K = A.shape
        else:
            (M, K), L = A.shape, 1
        N = B.shape[-1]
        sm_count = torch.cuda.get_device_properties(A.device).multi_processor_count
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # No device properties means we cannot justify a performance choice.
        # Keep the non-split baseline, which is always a legal fallback.
        return [conf for conf in configs if _split_k_of(conf) == 1]

    shortlisted = []
    for conf in configs:
        c = conf.kwargs["config"]
        if _split_k_of(conf) == 1:
            shortlisted.append(conf)
            continue
        cta_tile_m = cta_tile_shape_m(c.tile_m, c.cluster_m, device_capacity)
        tile_m, tile_n = (cta_tile_m, c.tile_n) if not c.swap_ab else (c.tile_n, cta_tile_m)
        ntiles = -(-M // tile_m) * -(-N // tile_n) * L
        k_tiles = -(-K // (c.tile_k or 64))
        split_k = _split_k_of(conf)
        if ntiles < sm_count and ntiles * split_k <= 4 * sm_count and 2 * split_k <= k_tiles:
            shortlisted.append(conf)
    return shortlisted


def prune_structural_gemm_configs(configs, named_args: dict, **kwargs):
    """Apply device/layout legality and dense split-K search-space expansion."""
    kwargs = named_args | kwargs
    gather_A = kwargs.get("A_idx") is not None
    varlen_m = kwargs.get("cu_seqlens_m") is not None
    varlen_k = kwargs.get("cu_seqlens_k") is not None
    device_capacity = get_device_capacity(kwargs["A"].device)[0]
    configs = [
        conf for conf in configs if conf.kwargs["config"].device_capacity == device_capacity
    ]
    configs = [
        conf
        for conf in configs
        if config_supports(conf.kwargs["config"], gather_A=gather_A, varlen_m=varlen_m)
    ]
    if not gather_A or device_capacity not in (10, 11):
        configs = [conf for conf in configs if not conf.kwargs["config"].use_tma_gather]
    if kwargs.get("SFA") is not None:
        configs = [conf for conf in configs if blockscaled_config_ok(conf.kwargs["config"])]
    if kwargs.get("SFD") is not None or kwargs.get("SFDCol") is not None:
        col_only = kwargs.get("SFDCol") is not None

        def sfd_ok(c: GemmConfig) -> bool:
            if c.swap_ab:
                return False
            if c.device_capacity in (10, 11):
                return (
                    c.tile_n % 64 == 0
                    and c.tile_m in (128, 256)
                    and not (c.tile_m == 128 and c.cluster_m % 2 == 0)
                )
            if c.device_capacity == 12:
                if col_only:
                    return c.tile_m % 64 == 0
                return c.tile_n % 32 == 0 and c.tile_m % 64 == 0 and c.tile_m % 192 != 0
            return False

        configs = [conf for conf in configs if sfd_ok(conf.kwargs["config"])]
    if (
        "split_k" in kwargs
        and kwargs["split_k"] is None
        and not (varlen_m or varlen_k or gather_A)
        and not (
            kwargs.get("SFA") is not None
            and _is_separate_split_k(kwargs.get("split_k_mode", 0))
        )
        and device_capacity in (9, 10, 11, 12)
    ):
        configs = _expand_split_k_configs(configs, kwargs["A"])
    return configs


class GemmTunePolicy(TunePolicy):
    """Generic GEMM policy: structural legality plus identity shortlist.

    Variable-length identity and measured selection are deliberately supplied
    by :class:`cute_gemm.varlen_selector.VarlenGemmTunePolicy`.  Keeping this
    parent dense-only makes a new GEMM operator inherit only the structural
    rules it actually shares.
    """

    def __init__(
        self,
    ) -> None:
        self.arg_names: tuple[str, ...] = ()
        self.arg_defaults: tuple[Any, ...] = ()

    def bind_signature(self, arg_names, arg_defaults) -> None:
        signature = (tuple(arg_names), tuple(arg_defaults))
        bound = getattr(self, "_bound_signature", None)
        if bound is not None:
            if bound != signature:
                raise ValueError("TunePolicy signature is already bound to a different function")
            return
        self._bound_signature = signature
        self.arg_names = tuple(arg_names)
        self.arg_defaults = tuple(arg_defaults)

    def make_key(self, args, kwargs, default_key):
        del args, kwargs
        return default_key()

    def budget_override(self, key):
        del key
        return False

    def prepare_candidates(self, configs, named_args, kwargs):
        return prune_structural_gemm_configs(configs, named_args, **kwargs)

    def shortlist(self, configs, named_args, kwargs):
        merged = dict(named_args)
        merged.update(kwargs)
        if merged.get("split_k") is not None:
            return configs
        if (
            merged.get("cu_seqlens_m") is not None
            or merged.get("cu_seqlens_k") is not None
            or merged.get("A_idx") is not None
        ):
            return [conf for conf in configs if _split_k_of(conf) == 1]
        A = merged.get("A")
        B = merged.get("B")
        if A is None or B is None:
            return configs
        try:
            device_capacity = get_device_capacity(A.device)[0]
        except (AssertionError, RuntimeError, TypeError, ValueError):
            # CPU/static policy tests and non-CUDA callers have no architecture
            # signal on which to justify a performance shortlist.
            return configs
        if device_capacity not in (9, 10, 11, 12):
            return [conf for conf in configs if _split_k_of(conf) == 1]
        return _shortlist_split_k_configs(configs, A, B, device_capacity)
