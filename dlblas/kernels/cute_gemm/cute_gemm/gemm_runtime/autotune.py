# Copyright (c) 2026, Tri Dao.
"""Generic autotuning for @gemm_epilogue mods.

``tuned_mod_gemm(mod, A, B, D, C, epi_args=...)`` sweeps the arch's GemmConfig
space directly through ``mod.gemm()`` with the existing Autotuner machinery
(CUDA-graph L2-cold bench, async-compile pool overlap, disk cache under
``$DLKERNEL_CACHE_DIR``) — no per-variant torch interface layer involved. Any mod
gets tuning for free; ``mod.gemm_tuned(...)`` is the method form.

Mechanics that make the generic path work with the Autotuner:

* One ``Autotuner`` per (mod semantic digest, epi-arg name set, has-C): the
  Autotuner derives its cache key and its L2-rotate clone sets from TOP-LEVEL
  named tensor kwargs, so ``epi_args`` is flattened into explicit kwargs and
  the wrapper carries a synthetic ``__signature__`` naming them (a dict value
  would neither key nor clone — every bench replay would share one buffer).
* ``mod_digest`` rides ``key=`` so editing the epilogue fn body invalidates
  in-memory AND disk tuning caches; the wrapper ``__name__`` embeds it so the
  ``<fn>.autotune.json`` files stay human-attributable.
* Reduce-sink buffers are tile-shaped ((l, m, n_tiles) / (l, m_tiles, n)):
  the entry point allocates missing buffers at the sweep's worst case
  (``sink_arg_shapes``), rejects undersized caller-owned buffers before the
  shared tuner key is consulted, then slices the full buffer per config. One
  buffer serves every tile size and the winning slice is returned in
  ``TunedModGemm.sinks``. A -1 fallback call has no winner, but it still
  consumes a slice: the wrapper stashes the views it ran with on the per-mod
  fn (``_last_fallback``) and ``tuned_mod_gemm`` returns them in ``sinks``
  while keeping ``config=None`` (the fallback-provenance contract).
* mod.gemm validation errors (ValueError/TypeError) are rewrapped into
  RuntimeError: the bench loop only converts RuntimeError/MemoryError into an
  inf timing, and a config a prune rule missed must not abort the sweep.
* Scalar epi args do not enter the tuning key (only tensor metadata does):
  tile choice is insensitive to scalar VALUES, and mod.gemm re-plans per
  metadata on the real call anyway.

varlen_m/gather_A tune through this path (cu_seqlens_m/A_idx are top-level
tensor kwargs, so they key and clone like any operand); swap_ab rides
swap-at-trace for element-mode sink-less mods; ``dynamic_scheduler=True``
forces dynamic-persistent scheduling on every candidate (matching the old
per-variant tuned wrappers); blockscaled SFA/SFB sweeps the
_blockscaled_ok-pruned space; ``concat_layout`` enters the tuner key (the
old per-variant tuners aliased concat/non-concat winners); A-operand
transforms tune through it too (the handle's semantic digest keys the tuner,
``transform_a.config_ok`` + geometry validation prune the space, and
runtime-operand bundles are rebuilt per config so their strip views bake the
candidate tiles). Not supported (yet): split_k.
"""

from __future__ import annotations

import inspect
from typing import NamedTuple

import torch

from cute_gemm.autotuner import AutotuneConfig, Autotuner
from cute_gemm.cute_dsl_utils import get_device_capacity
from cute_gemm.gemm_config import cta_tile_shape_m, get_all_configs

__all__ = ["TunedModGemm", "sink_arg_shapes", "tuned_mod_gemm"]


class TunedModGemm(NamedTuple):
    plan: object  # GemmEpiPlan of the winning call (already executed)
    config: object  # winning GemmConfig
    sinks: dict  # name -> the winning config's slice of the caller's buffer


def _cdiv(a, b):
    return (a + b - 1) // b


def _config_space(mod, device):
    """Coarse per-arch config list for this mod (before per-call pruning)."""
    cap = get_device_capacity(device)[0]
    hint = "gated" if mod.mode in ("acc_pair", "packed_cd_b16x2") else None
    cfgs = [
        c
        for c in get_all_configs(epilogue=hint)
        if c.device_capacity == cap
        # swap_ab rides swap-at-trace (2026-07-14): element-mode sink-less
        # mods only, enforced by GemmEpilogueTunePolicy host legality.
        and not (c.swap_ab and (mod.mode != "element" or mod.sinks))
        and not c.use_tma_gather  # gather_A untested through the fn frontend
        and (c.split_k is None or c.split_k == 1)  # split-K is default-epilogue-only
    ]
    if not cfgs:
        raise ValueError(f"no GemmConfigs for device capacity {cap}")
    return cfgs


def _gemm_mn(A, B, b_kn):
    n = B.shape[-1] if b_kn else B.shape[-2]
    m = A.shape[-2] if A.ndim == 3 else A.shape[0]
    return m, n


def _lead(A, A_idx, m_gemm):
    """(batch?, m) lead shape matching EpiMod._lead_shape (m_gemm already
    accounts for gather)."""
    return (A.shape[0], m_gemm) if A.ndim == 3 else (m_gemm,)


def _sink_slice(buf, shape):
    """The leading `shape` view of a worst-case sink buffer."""
    if tuple(buf.shape) == tuple(shape):
        return buf
    return buf[tuple(slice(0, s) for s in shape)]


def sink_arg_shapes(mod, m, n_gemm, l=None, device="cuda", num_seqs=None):
    """Worst-case (over the tuning sweep) buffer shapes for the mod's reduce
    sinks, keyed by sink name. Allocate these f32 and pass them in epi_args;
    the tuner slices per config and TunedModGemm.sinks returns the live view.
    ``num_seqs`` (varlen_m): dim==1 sinks size per-sequence tile-prefix rows."""
    cfgs = _config_space(mod, torch.device(device))
    min_tile_n = min(c.tile_n for c in cfgs)
    # blockscaled=False halves whenever the config could run 2-CTA — the
    # smallest possible per-CTA tile, so the buffer upper-bounds both modes.
    min_tile_m = min(cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity) for c in cfgs)
    lead = (m,) if l is None else (l, m)
    shapes = {}
    for name, op in mod.sinks.items():
        alloc = getattr(op, "sink_alloc_shape", None)
        if alloc is None:
            continue
        # cdiv is monotone in the tile size, so the min tiles give the sweep's
        # worst case (config-independent sinks ignore the tiles).
        shapes[name] = alloc(
            lead,
            n_gemm,
            min_tile_m,
            min_tile_n,
            num_seqs=num_seqs if getattr(op, "dim", 0) == 1 else None,
        )
    return shapes


def _validate_full_sink_buffers(mod, epi_args, A, B, *, b_kn, A_idx, cu_seqlens_m, n_gemm):
    """Enforce the shared-winner sink contract at the tuned-call boundary.

    A policy key deliberately describes workload metadata, not the capacity of
    a caller-owned scratch tensor. Every caller-supplied reduce sink therefore
    has to cover the worst tile geometry in the candidate space. Missing sinks
    are allocated here at that same worst-case shape; a larger caller
    allocation is accepted and sliced per candidate. A partial allocation is
    rejected before tuning or cache lookup so it can never poison a winner
    shared by another call in the same key bucket.
    """
    if not mod.sinks:
        return
    m_gemm = A_idx.shape[0] if A_idx is not None else _gemm_mn(A, B, b_kn)[0]
    l = A.shape[0] if A.ndim == 3 else None
    num_seqs = None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1
    required = sink_arg_shapes(
        mod,
        m_gemm,
        n_gemm,
        l=l,
        device=A.device,
        num_seqs=num_seqs,
    )
    for name, need in required.items():
        value = epi_args.get(name)
        if value is None:
            epi_args[name] = torch.empty(tuple(need), dtype=torch.float32, device=A.device)
            continue
        if not hasattr(value, "shape"):
            raise TypeError(f"sink '{name}' must be a tensor with shape at least {tuple(need)}")
        actual = tuple(value.shape)
        if len(actual) != len(need) or any(got < want for got, want in zip(actual, need)):
            raise ValueError(
                f"sink '{name}' is undersized for the tuned shared-winner contract: "
                f"need at least {tuple(need)}, got {actual}; allocate sink_arg_shapes(...)"
            )


def _slice_sinks(mod, epi_args, config, lead, n_gemm, blockscaled=False, num_seqs=None):
    views = {}
    for name, op in mod.sinks.items():
        alloc = getattr(op, "sink_alloc_shape", None)
        if alloc is None or name not in epi_args:
            continue
        cta_tile_m = cta_tile_shape_m(
            config.tile_m, config.cluster_m, config.device_capacity, blockscaled
        )
        views[name] = _sink_slice(
            epi_args[name],
            alloc(
                lead,
                n_gemm,
                cta_tile_m,
                config.tile_n,
                num_seqs=num_seqs if getattr(op, "dim", 0) == 1 else None,
            ),
        )
    return views


def _make_tuned_fn(mod, epi_names, transform_a=None, ta_names=()):
    sink_allocs = {
        n: op.sink_alloc_shape for n, op in mod.sinks.items() if hasattr(op, "sink_alloc_shape")
    }

    def fn(
        A=None,
        B=None,
        D=None,
        C=None,
        mod_digest=None,
        b_kn=False,
        cu_seqlens_m=None,
        A_idx=None,
        dynamic_scheduler=False,
        SFA=None,
        SFB=None,
        bs_format_a=None,
        bs_format_b=None,
        concat_layout=None,
        transform_digest=None,  # keyed; the mod itself is a closure capture
        transform_sf=None,
        config=None,
        tune_mode=None,  # rides through the Autotuner; the fn itself ignores it
        **epi_flat,  # epi args by name + transform operands as ta__<name>
    ):
        c = config
        # TuneMode fallback (the -1 call): no measured winner.  The miss
        # contract is the ANALYTICAL ranker (STATIC/RUNNING), not the mod's
        # blind default: a ranked config falls through to the explicit-tile
        # call below, which also slices reduce sinks for the KNOWN config
        # (worst-case buffers cannot honor a config-shaped sink contract).
        # Mirrors the epi __call__ non-tuner chain's guard: SM90, no gather;
        # dense stays uncovered by design.  (A is None only in signature
        # stubs; short-circuit keeps the device query real-tensor-only.)
        if c is None and A is not None and A_idx is None and get_device_capacity(A.device)[0] == 9:
            from cute_gemm import gemm_heuristic

            _desc = gemm_heuristic.desc_from_call(A, B, D, cu_seqlens_m, None)
            c = gemm_heuristic.rank(_desc, gemm_heuristic.sm90_candidates(_desc))
        if c is None and (sink_allocs or ta_names):
            # A config-shaped contract (reduce sinks, runtime-operand
            # transform bundles) cannot ride the blind default call: the
            # low-level gemm enforces EXACT sink shapes for its own tiling
            # and builds no operand bundle.  Resolve the mod's own default
            # config and take the explicit-tile path, which slices sinks and
            # builds bundles for a KNOWN config (the same chain the epi
            # non-tuner resolution uses).
            c = mod._default_config(A, B, transform_a)
        if c is None:
            # Uncovered family with no config-shaped contract (no sinks, no
            # runtime-operand transform): the mod's own default-config path
            # resolves the tiling.  Delegates to the LOW-LEVEL mod.gemm
            # (tile-parameter method, no tuner entry) — the -1 execution is
            # loop-free by construction; it must not gain tune_mode/config
            # kwargs (it has neither).  Stash the RAW epi_args entries for the
            # sink names (an empty dict here) so the every-fallback-call
            # invariant of _last_fallback holds for tuned_mod_gemm's read.
            fn._last_fallback = (
                None,
                {name: epi_flat[name] for name in sink_allocs if name in epi_flat},
            )
            return mod.gemm(
                A,
                B,
                D,
                C,
                epi_args={name: epi_flat[name] for name in epi_names},
                b_kn=b_kn,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                concat_layout=concat_layout,
                transform_a=transform_a,
                transform_sf=transform_sf,
            )
        m_gemm, n_gemm = _gemm_mn(A, B, b_kn)
        if transform_a is not None and transform_a.padded_n(B) is not None:
            n_gemm = transform_a.padded_n(B)  # B is the repacked blob
        if A_idx is not None:
            m_gemm = A_idx.shape[0]
        lead = _lead(A, A_idx, m_gemm)
        num_seqs = None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1
        cta_tile_m = cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity, SFA is not None)
        epi_args = {}
        for name in epi_names:
            v = epi_flat[name]
            alloc = sink_allocs.get(name)
            if alloc is not None and isinstance(v, torch.Tensor):
                op_dim = getattr(mod.sinks.get(name), "dim", 0)
                v = _sink_slice(
                    v,
                    alloc(
                        lead,
                        n_gemm,
                        cta_tile_m,
                        c.tile_n,
                        num_seqs=num_seqs if op_dim == 1 else None,
                    ),
                )
            epi_args[name] = v
        if config is None:
            # Ranked -1 fall-through: report THIS config's sliced sink views
            # to tuned_mod_gemm.  The stash lives on the per-mod fn (one fn
            # per mod per tuner key) and is set on EVERY fallback call —
            # never on sweep/winner/pinned calls, which carry an explicit
            # config; tuned_mod_gemm reads it immediately after its own -1
            # call returns, so the last write is the call that produced its
            # plan.
            fn._last_fallback = (
                c,
                {name: epi_args[name] for name in sink_allocs if name in epi_args},
            )
        dyn = c.is_dynamic_persistent or dynamic_scheduler
        # SM90 dynamic-persistent scheduling consumes a semaphore; a fresh
        # zeros(1) per call is the gemm_interface pattern (under the CUDA-graph
        # bench the captured memset re-zeros it on every replay).
        sem = None
        if dyn and get_device_capacity(A.device)[0] == 9:
            sem = torch.zeros(1, dtype=torch.int32, device=A.device)
        B_pass, bkn_pass = B, b_kn
        if c.swap_ab and not b_kn:
            B_pass, bkn_pass = B.mT, True  # swap_ab requires B given (k, n)
        try:
            A_pass = A
            if transform_a is not None and transform_a.needs_operands:
                # per-config bundle: the strip views bake this config's tiles
                # (a geometry mismatch with the caller's strips raises here —
                # pre-compile — and benches as inf)
                A_pass = transform_a.bundle(
                    A, {n: epi_flat[f"ta__{n}"] for n in ta_names}, c.tile_m, c.tile_k
                )
            return mod.gemm(
                A_pass,
                B_pass,
                D,
                C,
                epi_args=epi_args,
                tile_M=c.tile_m,
                tile_N=c.tile_n,
                tile_K=None if SFA is not None else c.tile_k,
                cluster_M=c.cluster_m,
                cluster_N=c.cluster_n,
                pingpong=c.pingpong,
                is_dynamic_persistent=dyn,
                max_swizzle_size=c.max_swizzle_size,
                tile_count_semaphore=sem,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                concat_layout=concat_layout,
                b_kn=b_kn,
                swap_ab=c.swap_ab,
                transform_a=transform_a,
                transform_sf=transform_sf,
            )
        except (ValueError, TypeError, AssertionError) as e:
            # The bench loop only maps RuntimeError/MemoryError to an inf
            # timing; a config the prune missed must not abort the sweep.
            raise RuntimeError(f"config {c} rejected: {e}") from e

    fn.__name__ = f"mod_{mod._ident}"
    kw = inspect.Parameter.KEYWORD_ONLY
    params = [
        inspect.Parameter(n, kw, default=None)
        for n in (
            "A",
            "B",
            "D",
            "C",
            "mod_digest",
            "cu_seqlens_m",
            "A_idx",
            "SFA",
            "SFB",
            "bs_format_a",
            "bs_format_b",
            "config",
        )
    ]
    params.append(inspect.Parameter("b_kn", kw, default=False))
    params.append(inspect.Parameter("dynamic_scheduler", kw, default=False))
    params.append(inspect.Parameter("tune_mode", kw, default=None))
    params.append(inspect.Parameter("concat_layout", kw, default=None))
    params.append(inspect.Parameter("transform_digest", kw, default=None))
    params.append(inspect.Parameter("transform_sf", kw, default=None))
    params.extend(inspect.Parameter(f"ta__{n}", kw, default=None) for n in ta_names)
    params.extend(inspect.Parameter(n, kw, default=None) for n in epi_names)
    fn.__signature__ = inspect.Signature(params)
    return fn


_MOD_TUNERS: dict = {}


def _get_tuner(mod, epi_names, has_c, device, transform_a=None, ta_names=()):
    key = (
        mod.semantic_digest,
        epi_names,
        has_c,
        get_device_capacity(device)[0],
        getattr(transform_a, "semantic_digest", None),
        ta_names,
    )
    tuner = _MOD_TUNERS.get(key)
    if tuner is None:
        from cute_gemm.gemm_epilogue_tune_policy import GemmEpilogueTunePolicy

        tuner = Autotuner(
            _make_tuned_fn(mod, epi_names, transform_a, ta_names),
            # key= stays the dense default key's only scalar channel; the
            # policy's varlen bucket key carries them as named scalars.
            key=[
                "mod_digest",
                "b_kn",
                "dynamic_scheduler",
                "concat_layout",
                "bs_format_a",
                "bs_format_b",
                "transform_digest",
            ],
            configs=[AutotuneConfig(config=c) for c in _config_space(mod, device)],
            policy=GemmEpilogueTunePolicy(
                mod,
                transform_a,
            ),
            cache_results=True,
        )
        _MOD_TUNERS[key] = tuner
    return tuner


def tuned_mod_gemm(
    mod,
    A,
    B,
    D,
    C=None,
    *,
    epi_args,
    b_kn=False,
    cu_seqlens_m=None,
    A_idx=None,
    dynamic_scheduler=False,
    SFA=None,
    SFB=None,
    bs_format_a=None,
    bs_format_b=None,
    concat_layout=None,
    # A-operand transform: the handle keys the tuner (semantic digest);
    # layout-owning transforms pass B as the repacked blob (+ transform_sf),
    # runtime-operand transforms pass RAW operand tensors (bundles are built
    # per config inside the sweep — their strip views bake the tiles).
    transform_a=None,
    transform_sf=None,
    transform_operands=None,
    tune_mode=None,
):
    """Autotuned ``mod.gemm`` with a full-buffer sink contract.

    The first call sweeps the arch's config space per (mod, tensor metadata),
    then warm calls replay the winner through ``mod.gemm``'s plan cache.
    Missing reduce sinks are allocated at the sweep's worst case (see
    ``sink_arg_shapes``); undersized caller-owned buffers are rejected before
    cache lookup.
    Returns ``TunedModGemm(plan, config, sinks)`` with the winning config and
    its sink views.  A -1 fallback (no measured winner) keeps ``config=None``
    — the fallback-provenance signal — and ``sinks`` carries the views the
    delegated call actually consumed.
    """
    # Keep caller-owned mappings immutable while allowing the tuner to fill in
    # missing reduce sinks with its full worst-case scratch allocation.
    epi_args = dict(epi_args)
    if transform_a is not None:
        from cute_gemm.operand_transform.host import as_transform_mod

        transform_a = as_transform_mod(transform_a)
    m_gemm, n_gemm = _gemm_mn(A, B, b_kn)
    if transform_a is not None and transform_a.padded_n(B) is not None:
        n_gemm = transform_a.padded_n(B)
    flipped = transform_a is not None and transform_a.owned_fmt is not None
    mod._preflight_colvecs(
        epi_args,
        rows=mod._colvec_rows(A, A_idx, n_gemm, flipped),
        varlen_m=cu_seqlens_m is not None,
        flipped=flipped,
    )
    _validate_full_sink_buffers(
        mod,
        epi_args,
        A,
        B,
        b_kn=b_kn,
        A_idx=A_idx,
        cu_seqlens_m=cu_seqlens_m,
        n_gemm=n_gemm,
    )
    epi_names = tuple(sorted(epi_args))
    ta_names = tuple(sorted(transform_operands)) if transform_operands else ()
    assert not any(f"ta__{n}" in epi_args for n in ta_names)
    tuner = _get_tuner(mod, epi_names, C is not None, A.device, transform_a, ta_names)
    plan = tuner(
        A=A,
        B=B,
        D=D,
        C=C,
        mod_digest=mod.semantic_digest,
        b_kn=b_kn,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        concat_layout=concat_layout,
        transform_digest=getattr(transform_a, "semantic_digest", None),
        transform_sf=transform_sf,
        tune_mode=tune_mode,
        **{f"ta__{k}": v for k, v in (transform_operands or {}).items()},
        **epi_args,
    )
    best_obj = tuner.best_config
    if best_obj is None:
        # -1 fallback resolution: the delegated call already executed with the
        # config the wrapper resolved (analytical ranker, or the mod's own
        # default tiling when the family is uncovered).  ``config`` stays None
        # -- callers read it as fallback provenance -- but the wrapper stashed
        # the exact sink views this call consumed, so the caller's owned-sink
        # hand-off sees real buffers instead of a KeyError.  The stash is
        # per-fn and rewritten by every fallback call; this read immediately
        # follows this call's own -1 execution.
        _, fallback_sinks = tuner.fn._last_fallback
        return TunedModGemm(plan, None, fallback_sinks)
    best = best_obj.kwargs["config"]
    if A_idx is not None:
        m_gemm = A_idx.shape[0]
    return TunedModGemm(
        plan,
        best,
        _slice_sinks(
            mod,
            epi_args,
            best,
            _lead(A, A_idx, m_gemm),
            n_gemm,
            SFA is not None,
            num_seqs=None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1,
        ),
    )
