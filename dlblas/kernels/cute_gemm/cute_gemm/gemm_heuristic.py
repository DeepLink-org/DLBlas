# Copyright (c) 2026, Tri Dao.
"""Analytical fallback config selection (the TuneMode ``-1`` interpreter).

``rank(desc, candidates) -> GemmConfig | None`` picks a legal config for a
call that has no measured winner, using a deterministic wave-and-traffic
model over the operator's own prepared candidate pool.  It never measures,
never queries CUDA (SM count is resolved at the call boundary and carried in
the descriptor), never reads tensor values, and never invents candidates.

Design contract (plan v4 §3.1):

* candidates come from the operator's existing prepare phase
  (``sm90_candidates`` below reuses the structural prune, including the
  varlen-K dynamic twins via :class:`VarlenGemmTunePolicy`), so the ranker
  can only reorder, never widen, the legal space;
* the cost terms are explainable — one DeepGEMM-shaped expression per
  family (bandwidth-normalized L1/L2 traffic over wave efficiency), with
  every constant carrying its unit and its measurement provenance: the
  formula below was frozen 2026-10-21 from the 1584-case offline study
  (``.vscode/static_tune_study``; holdout protocol and numbers in
  ``gemm_static_tune_master.md`` 卷二/卷三);
* DeepGEMM's hard exclusion of single-wave multicast layouts is
  deliberately NOT copied: our candidate grid differs, and small shapes
  must resolve through the same finite lexicographic order as everything
  else;
* the final tie-break is a total order over config fields, so the pick is
  independent of the candidate list's order.

Uncovered (arch, family) combinations return ``None``; the caller then uses
its existing default-config chain (nvMMH / ``default_config``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from cute_gemm.autotuner import AutotuneConfig, _tune_trace
from cute_gemm.gemm_config import GemmConfig, cta_tile_shape_m, get_all_configs

__all__ = ["GemmHeuristicDesc", "desc_from_call", "rank", "sm90_candidates", "sm_count_for"]

#: Bytes moved per element by dtype width.  Initial unfitted constant: the
#: model treats L1/L2/HBM widths as uniform per byte; R3 may split them.
_ELEM_BYTES_PER_BIT = 1 / 8

#: SM count fallback per device capacity when neither the environment nor a
#: device can answer (CPU-only prefill).  A wrong guess only selects a
#: different tile, never a wrong result.
_SM_COUNT_FALLBACK = {9: 132, 10: 148, 11: 148, 12: 170, 8: 108}


def sm_count_for(device_capacity: int, device=None) -> int:
    """SM count for ranking: ``DLKERNEL_SM_COUNT`` env, then the call's own
    device (not device 0 — heterogeneous boxes would otherwise answer for
    the wrong GPU), then a per-arch constant.  Resolved once at the call
    boundary and passed explicitly into the descriptor — ``rank`` itself
    never queries."""
    env = os.environ.get("DLKERNEL_SM_COUNT")
    if env is not None:
        return int(env)
    try:
        from cute_gemm.cute_dsl_utils import get_num_sms

        idx = getattr(device, "index", None)
        # get_num_sms honours the set_num_sms override (DeepGEMM semantics)
        # and falls back to the device's own count when unset.
        return get_num_sms(idx if idx is not None else torch.cuda.current_device())
    except Exception:  # noqa: BLE001 -- deliberate: GPU-blind callers take the constant
        return _SM_COUNT_FALLBACK.get(device_capacity, 132)


@dataclass(frozen=True)
class GemmHeuristicDesc:
    """Immutable ranking input: math dimensions plus explicit hardware
    parameters.  ``m``/``n``/``k`` are the *dense* math dims of one logical
    GEMM; grouped families carry the group count in ``l`` (and ``experts``
    for envelope reporting).  No tensors, no device queries, no values."""

    family: str  # "dense" | "varlen_m" | "varlen_k" | "epi"
    m: int
    n: int
    k: int
    l: int = 1
    dtype_bits: int = 16
    cd_bits: int = 16
    device_capacity: int = 9
    sm_count: int = 132
    experts: int = 0


class _ArchHeuristic:
    """Per-arch base: every family raises, subclasses override what they
    implement — an unimplemented family is simply an uncovered combination."""

    def rank_dense(self, desc, candidates):
        raise NotImplementedError

    def rank_varlen_m(self, desc, candidates):
        raise NotImplementedError

    def rank_varlen_k(self, desc, candidates):
        raise NotImplementedError

    def rank_epi(self, desc, candidates):
        raise NotImplementedError


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# =========================================================================== #
# SM90 cost formula — v4, frozen 2026-10-21 from the 1584-case offline study
# (ablation in .vscode/static_tune_study/v4_fit.py; every step measured).
# =========================================================================== #
#
#     T(c) = L1_CYCLES(c) / wave_eff(c)
#
#     L1_CYCLES = (k_blk * ((tm + tn) + (max(64, tm) + tn)) * elem
#                  + 2 * tm*tn*cd_elem) * blocks / l1_bw
#                 # full three-term form of DeepGEMM sm90.hpp:222-230:
#                 # l1_ab + l1_tc(+cd) + l1_l2_cd — same terms, our kernel.
#     l1_bw     = 128 B/clk/SM (sm90.hpp:213; HBM/TC are config-invariant)
#     wave_eff  = blocks / (waves * sm_count)   # waves over CTAs, DeepGEMM
#                 # sm90.hpp:207/231 form — blocks are CTA tiles, the divisor
#                 # is the SM count (NOT sm_count//cluster, which counts
#                 # clusters; an earlier dimensional slip cost m ~1.5pp).
#     varlen_k dynamic-persistent candidates: wave_eff' = min(1, +0.1).
#                 # A SWITCH, not a fitted constant: credit 0 is a cliff
#                 # (k 14.2%), 0.1/0.2/0.5 are indistinguishable (47.9%+).
#                 # Mechanism: grouped-k work-queue balancing, not tail-wave
#                 # recovery (proportional-tail form fit to zero).
#
# The L2 multicast-amortization term was REMOVED in v4: with the fitted
# weights it won 0 of 47,036 (case, config) pairs — dead code.  The observed
# cluster (1,2) preference comes from the tie-break order, not from physics
# inside the formula (see the tie tuple below).
#
# Cell prune (2-D post-swap output tile, K=1 per cell — the "prune model" of
# DeepGEMM's enumerate->prune->compare, thresholds frozen from data, 4-fold
# consistent; avg = ragged-dim rows per expert, uniform-routing model —
# cu_seqlens VALUES are device-resident, only the expert COUNT is visible):
#
#     varlen_m: avg = m//E;  <128 -> (tm,tn)=(128,256); <8192 -> (256,128);
#                              else (192,128)
#     varlen_k: avg = k//E;  <170 -> (128,256);         <2048 -> (128,128);
#                              else (256,128)
#     (tm,tn) are the POST-SWAP effective output tile.  An empty cell (pool
#     drift) falls back to the full pool — the fallback never fails.
#
# Holdout (4-fold leave-3-models-out): varlen_m top-5 80.9% (<=1.05 78.1%,
# median 1.010); varlen_k 59.5% (<=1.05 75.8%, median 1.021).  Best measured
# alternative form (data tree): 78.9% / 61.2% — v4 matches it for m and sits
# 1.7pp below for k while staying one readable expression.  Dense is
# intentionally NOT covered (nvMMH owns that path).

_DYN_WAVE_CREDIT = 0.1
_WGMMA_M = 64
#: (t1, t2) cell boundaries and the three (post-swap tm, tn) cells they select.
_CELLS = {
    "varlen_m": ((128, 8192), ((128, 256), (256, 128), (192, 128))),
    "varlen_k": ((170, 2048), ((128, 256), (128, 128), (256, 128))),
}


def _ragged_avg(desc: GemmHeuristicDesc) -> int:
    """Rows per expert on the family's ragged dim (uniform-routing model)."""
    ragged = desc.m if desc.family == "varlen_m" else desc.k
    return max(ragged // max(desc.experts, 1), 1)


def _effective_tile(c: GemmConfig):
    """Post-swap output tile (tm, tn): swap_ab exchanges the operands, so the
    tile the scheduler sees on M is ``tile_n`` when swapped."""
    cta_m = cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity)
    return (cta_m, c.tile_n) if not c.swap_ab else (c.tile_n, cta_m)


def _cell(desc: GemmHeuristicDesc):
    (t1, t2), cells = _CELLS[desc.family]
    avg = _ragged_avg(desc)
    return cells[0] if avg < t1 else (cells[1] if avg < t2 else cells[2])


def _formula_cost(desc: GemmHeuristicDesc, c: GemmConfig):
    """The frozen expression above as a sort key (total order via ``tie``).

    Family dimensionality (v3 correction, externally corroborated by
    DeepGEMM's KGroupedContiguous expected_k/m/groups): varlen_m's ragged dim
    TILES the output and every block walks the full K; varlen_k's ragged dim
    IS the contraction — it sets the per-block mainloop length (k_e = k//E)
    and does NOT tile outputs.
    """
    tm, tn = _effective_tile(c)
    E = max(desc.experts, 1)
    if desc.family == "varlen_k":
        k_blk = _ragged_avg(desc)
        blocks = E * _cdiv(desc.m, tm) * _cdiv(desc.n, tn)
    else:
        k_blk = desc.k
        blocks = E * _cdiv(_ragged_avg(desc), tm) * _cdiv(desc.n, tn)
    waves = _cdiv(blocks, desc.sm_count)
    wave_eff = blocks / (waves * desc.sm_count)
    if desc.family == "varlen_k" and c.is_dynamic_persistent:
        wave_eff = min(1.0, wave_eff + _DYN_WAVE_CREDIT)
    eb = desc.dtype_bits * _ELEM_BYTES_PER_BIT
    cd = tm * tn * desc.cd_bits * _ELEM_BYTES_PER_BIT
    l1 = (k_blk * ((tm + tn) + (max(_WGMMA_M, tm) + tn)) * eb + 2 * cd) * blocks
    cost = l1 / (128 * desc.sm_count) / wave_eff
    tie = (
        c.tile_m,
        c.tile_n,
        c.cluster_m,
        c.cluster_n,
        int(bool(c.swap_ab)),
        int(bool(c.pingpong)),
        int(bool(c.is_dynamic_persistent)),
        c.tile_k or 0,
    )
    return (cost,) + tie


class _Sm90Heuristic(_ArchHeuristic):
    device_capacity = 9

    @staticmethod
    def _pick(desc, candidates):
        want = _cell(desc)
        scored = [
            (_formula_cost(desc, c), i, c, _effective_tile(c)) for i, c in enumerate(candidates)
        ]
        in_cell = [t for t in scored if t[3] == want] or scored
        return min(in_cell, key=lambda t: (t[0], t[1]))[2]

    def rank_varlen_m(self, desc, candidates):
        return self._pick(desc, candidates)

    def rank_varlen_k(self, desc, candidates):
        return self._pick(desc, candidates)


_SM90 = _Sm90Heuristic()


def rank(desc: GemmHeuristicDesc, candidates: Sequence[GemmConfig]) -> GemmConfig | None:
    """Best legal config for ``desc`` within ``candidates`` (reorder-only).

    Returns ``None`` for an uncovered (arch, family) combination or an empty
    candidate list; the caller then falls back to its default-config chain.
    """
    impl = _SM90 if desc.device_capacity == 9 else None
    if impl is None or not candidates:
        return None
    method = getattr(impl, f"rank_{desc.family}", None)
    if method is None:
        return None
    try:
        return method(desc, candidates)
    except NotImplementedError:
        return None
    except Exception as exc:  # noqa: BLE001 -- the fallback degrades, never breaks the call
        _tune_trace(
            f"HEURISTIC_RANK_ERROR family={desc.family} arch={desc.device_capacity} err={exc!r}"
        )
        return None


@lru_cache(maxsize=32)
def _prepared_pool_cache(device_capacity: int, varlen_k: bool, varlen_m: bool):
    """The operator's structural prepare result for a metadata signature,
    reusing the shared prune (single statement of legality).

    GPU-blind callers (CPU tests, prefill) run under ``DLKERNEL_ARCH``, the
    repository's existing cross-compile override, so the prune's capability
    lookup never touches a device; on a real box the cpu/meta fallback in
    ``_get_device_capacity_cached`` auto-detects the local card.
    ``split_k=1`` keeps the expansion off: the fallback ranks the base pool,
    matching a pinned-config caller.
    """
    from cute_gemm.varlen_selector import VarlenGemmTunePolicy

    configs = [
        AutotuneConfig(config=c) for c in get_all_configs() if c.device_capacity == device_capacity
    ]
    kwargs = {
        "A_idx": None,
        "cu_seqlens_m": True if varlen_m else None,
        "cu_seqlens_k": True if varlen_k else None,
        "SFA": None,
        "SFD": None,
        "SFDCol": None,
        "split_k": 1,
    }
    pruned = VarlenGemmTunePolicy().prepare_candidates(
        configs, {"A": _MetaTensor(), "B": _MetaTensor()}, kwargs
    )
    return tuple(conf.kwargs["config"] for conf in pruned)


class _MetaTensor:
    """Minimal operand stand-in: the prune reads only metadata, and the
    capability lookup is satisfied by ``DLKERNEL_ARCH``."""

    shape = (1, 1)
    dtype = 16

    @property
    def device(self):
        import torch

        return torch.device("cpu")


def sm90_candidates(desc: GemmHeuristicDesc) -> tuple:
    """The SM90 structural pool for desc's family (dense / varlen_m /
    varlen_k), via the shared prune — including the varlen-K dynamic twins."""
    return _prepared_pool_cache(
        desc.device_capacity,
        varlen_k=desc.family == "varlen_k",
        varlen_m=desc.family == "varlen_m",
    )


def desc_from_call(A, B, out, cu_seqlens_m=None, cu_seqlens_k=None) -> GemmHeuristicDesc:
    """Descriptor from a call's tensor *metadata* only (no values, no D2H).

    ``m``/``k`` come from A's trailing dims, ``n`` from B's last dim (the
    ``gemm_tuned`` (K, N) convention), ``l`` from A's optional batch axis.
    Grouped families additionally carry the expert count.  SM count is
    resolved here — the call boundary — exactly once per capacity.
    """
    cu = cu_seqlens_m if cu_seqlens_m is not None else cu_seqlens_k
    family = (
        "varlen_m"
        if cu_seqlens_m is not None
        else "varlen_k"
        if cu_seqlens_k is not None
        else "dense"
    )
    from cute_gemm.cute_dsl_utils import get_device_capacity

    device_capacity = get_device_capacity(A.device)[0]
    return GemmHeuristicDesc(
        family=family,
        m=A.shape[-2],
        n=B.shape[-1],
        k=A.shape[-1],
        l=A.shape[0] if A.ndim == 3 else 1,
        dtype_bits=A.element_size() * 8,
        cd_bits=(out.element_size() if out is not None else A.element_size()) * 8,
        device_capacity=device_capacity,
        sm_count=sm_count_for(device_capacity, A.device),
        experts=0 if cu is None else cu.shape[0] - 1,
    )
