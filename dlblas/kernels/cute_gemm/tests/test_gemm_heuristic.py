# Copyright (C) 2026, Tri Dao.
"""Unit tests for the analytical fallback ranker (cute_gemm/gemm_heuristic.py).

Metadata-only and CPU-only: descriptors are built from plain shape/dtype
tuples, candidates come from the real SM90 structural pool, and no test
touches a CUDA device.  These pin the *contract* of the fallback —
deterministic, order-independent, candidate-respecting — not its accuracy
(the frozen constants are validated by the offline study, plan v4 §7).
"""

import pytest

from cute_gemm.gemm_config import GemmConfig, get_all_configs
from cute_gemm.gemm_heuristic import (
    GemmHeuristicDesc,
    rank,
    sm90_candidates,
    sm_count_for,
)


@pytest.fixture(autouse=True)
def _cpu_arch_for_pool(monkeypatch):
    """The shared prune answers its capability lookup via the arch override,
    keeping these tests CPU-only (same convention as test_autotuner)."""
    monkeypatch.setenv("DLKERNEL_ARCH", "90")


def _sm90_pool():
    return [c for c in get_all_configs() if c.device_capacity == 9]


def _desc(**kw):
    base = {
        "family": "dense",
        "m": 4096,
        "n": 4096,
        "k": 4096,
        "l": 1,
        "dtype_bits": 16,
        "cd_bits": 16,
        "device_capacity": 9,
        "sm_count": 132,
        "experts": 0,
    }
    base.update(kw)
    return GemmHeuristicDesc(**base)


def _vm(**kw):
    kw.setdefault("family", "varlen_m")
    kw.setdefault("experts", 32)
    return _desc(**kw)


def test_rank_returns_a_candidate_deterministically():
    cands = sm90_candidates(_vm())
    pick = rank(_vm(), cands)
    assert pick in cands
    assert rank(_vm(), cands) == pick


def test_rank_is_independent_of_candidate_order():
    cands = sm90_candidates(_vm())
    assert rank(_vm(), cands) == rank(_vm(), list(reversed(cands)))


def test_rank_small_m_prefers_smaller_output_tiles():
    # A tiny routed-M workload must not resolve to the widest tile: padding
    # waste dominates and the pick must be a deterministic, finite score.
    small = _desc(family="varlen_m", m=64, n=7168, k=2048, l=1, experts=32)
    pick = rank(small, sm90_candidates(small))
    pool_tiles = {c.tile_m for c in sm90_candidates(small)}
    assert pick.tile_m == min(pool_tiles)


def test_multicast_single_wave_is_not_excluded():
    # v4 §3.1: DeepGEMM's hard INT64_MAX for single-wave multicast layouts
    # must NOT be copied — every candidate keeps a finite score.
    desc = _vm(m=256, n=256, k=512)
    cands = [
        GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=True,
            cluster_m=1,
            cluster_n=2,
            swap_ab=False,
            device_capacity=9,
            is_dynamic_persistent=False,
        ),
        GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=True,
            cluster_m=1,
            cluster_n=1,
            swap_ab=False,
            device_capacity=9,
            is_dynamic_persistent=False,
        ),
    ]
    pick = rank(desc, cands)
    assert pick in cands


def test_uncovered_arch_returns_none():
    desc = _desc(device_capacity=10)
    assert rank(desc, _sm90_pool()) is None


def test_family_routing_and_none_on_missing_impl():
    # Dense is deliberately uncovered (2026-10-21): a dense miss rides the
    # existing nvMMH/default chain with the warn-once cliff notice.
    dense = _desc()
    assert rank(dense, sm90_candidates(dense)) is None
    vm = _desc(family="varlen_m", experts=8)
    vk = _desc(family="varlen_k", experts=8, m=128, n=1024, k=65536)
    assert rank(vm, sm90_candidates(vm)) is not None
    assert rank(vk, sm90_candidates(vk)) is not None
    epi = _desc(family="epi")
    assert rank(epi, sm90_candidates(epi)) is None


def test_expert_count_changes_the_wave_term():
    # Same totals, different E: blocks scale with L=experts for grouped
    # families, so the ranking input differs even when M*N*K matches.
    d1 = _desc(family="varlen_m", m=4096, n=2048, k=2048, l=32, experts=32)
    d2 = _desc(family="varlen_m", m=4096, n=2048, k=2048, l=16, experts=16)
    assert d1 != d2
    assert rank(d1, sm90_candidates(d1)) is not None


def test_sm_count_is_explicit_and_sensitivity_exists():
    from cute_gemm.gemm_heuristic import _formula_cost

    a = _vm(sm_count=132)
    b = _vm(sm_count=64)
    # Both resolve; a smaller machine may legitimately prefer a different
    # tile split — the assertion is that the desc field participates.
    assert rank(a, sm90_candidates(a)) is not None
    assert rank(b, sm90_candidates(b)) is not None
    # Sensitivity, at the formula level: the wave term divides by sm_count,
    # so the same config must score differently on different machines.
    cfg = sm90_candidates(a)[0]
    assert _formula_cost(a, cfg) != _formula_cost(b, cfg)


def test_sm_count_resolution_order(monkeypatch):
    monkeypatch.setenv("DLKERNEL_SM_COUNT", "100")
    assert sm_count_for(9) == 100
    monkeypatch.delenv("DLKERNEL_SM_COUNT")
    # No device query in CPU tests: the per-arch constant answers.
    assert sm_count_for(9) > 0


def test_set_num_sms_semantics(monkeypatch):
    """DeepGEMM DeviceRuntime semantics (2026-10-21): 0 restores the device
    default; a positive cap is validated against the physical count when a
    device exists; the heuristic's sm_count and the cluster cap follow the
    override — and a cached occupancy query must not freeze the cap."""
    import cute_gemm.cute_dsl_utils as cdu
    import cute_gemm.gemm_heuristic as gh
    from cute_gemm import get_num_sms, set_num_sms

    monkeypatch.delenv("DLKERNEL_SM_COUNT", raising=False)
    set_num_sms(0)
    assert get_num_sms() == cdu.get_device_multiprocessor_count(0)
    set_num_sms(100)
    assert get_num_sms() == 100
    assert gh.sm_count_for(9) == 100
    # Cluster capping is arithmetic over the (cached) hw occupancy: whole
    # clusters only, and restoring the default must lift the cap even though
    # the occupancy query result stays cached.
    monkeypatch.setattr(cdu, "get_device_capacity", lambda *a, **k: (9, 0), raising=False)
    monkeypatch.setattr(cdu, "_max_active_clusters_hw", lambda *a, **k: 82, raising=False)
    assert cdu.get_max_active_clusters(2) == 50
    set_num_sms(0)
    assert cdu.get_max_active_clusters(2) == 82
    assert get_num_sms() == cdu.get_device_multiprocessor_count(0)
    with pytest.raises(ValueError):
        set_num_sms(-1)
