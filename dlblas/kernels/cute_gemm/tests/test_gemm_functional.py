# Copyright (c) 2026, Tri Dao.
"""Functional facade ops (dlkernel::gemm, dlkernel::gemm_add, dlkernel::gemm_epi_f):
allocation inside the op, real fakes — one graph-insertable node per call.
Parity vs the python wrappers, and fake-tensor shape/dtype correctness (the
property FX passes rely on)."""

import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cute_gemm.gemm_interface import gemm, gemm_add
from cute_gemm import TuneMode

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="needs sm90",
)


def _abc(m=512, k=512, n=512, dt=torch.bfloat16, c_dtype=None):
    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=dt, device="cuda") / math.sqrt(k)
    B = torch.randn(k, n, dtype=dt, device="cuda") / math.sqrt(k)
    C = torch.randn(m, n, dtype=c_dtype or dt, device="cuda")
    return A, B, C


@requires_sm90
def test_gemm_add_functional_parity(tmp_path, monkeypatch):
    # STATIC still consults the tune cache (heuristic fallback); keep it out of $HOME.
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    A, B, C = _abc()
    bias = torch.randn(B.shape[-1], dtype=A.dtype, device="cuda")
    out = torch.ops.dlkernel.gemm_add(A, B, C, bias=bias, tune_mode=TuneMode.STATIC)
    ref = gemm_add(A, B, C, bias=bias, tune_mode=TuneMode.STATIC)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_gemm_add_functional_fp32_residual(tmp_path, monkeypatch):
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    A, B, C = _abc(c_dtype=torch.float32)
    out = torch.ops.dlkernel.gemm_add(A, B, C, out_dtype=torch.float32, tune_mode=TuneMode.STATIC)
    assert out.dtype == torch.float32
    ref = gemm_add(A, B, C, out_dtype=torch.float32, tune_mode=TuneMode.STATIC)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_gemm_functional_parity(tmp_path, monkeypatch):
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    A, B, _ = _abc()
    out = torch.ops.dlkernel.gemm(A, B, tune_mode=TuneMode.STATIC)
    ref = gemm(A, B, tune_mode=TuneMode.STATIC)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_functional_fakes():
    """Fake kernels must report the shapes/dtypes FX passes will rely on,
    without touching CUDA."""
    with FakeTensorMode():
        A = torch.empty(64, 32, dtype=torch.bfloat16, device="cuda")
        B = torch.empty(32, 128, dtype=torch.bfloat16, device="cuda")
        C = torch.empty(64, 128, dtype=torch.float32, device="cuda")
        out = torch.ops.dlkernel.gemm_add(A, B, C, out_dtype=torch.float32)
        assert out.shape == (64, 128) and out.dtype == torch.float32
        out2 = torch.ops.dlkernel.gemm(A, B)
        assert out2.shape == (64, 128) and out2.dtype == torch.bfloat16


@requires_sm90
@pytest.mark.parametrize("mode", list(TuneMode))
def test_gemm_fullgraph_modes_match_eager(mode, tmp_path, monkeypatch):
    """C8 (plan v4 §6): every TuneMode under fullgraph matches the eager
    reference — the mode survives the compile boundary, no graph break."""
    # TUNED/RUNNING sweep and persist a winner; STATIC/REQUIRE consult the
    # cache. Keep every mode's disk traffic inside pytest's tmp_path.
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    A, B, _ = _abc()
    ref = gemm(A, B, tune_mode=mode)
    fn = torch.compile(
        lambda a, b: torch.ops.dlkernel.gemm(a, b, tune_mode=int(mode)), fullgraph=True
    )
    out = fn(A, B)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@requires_sm90
def test_fake_trace_never_tunes_or_consumes_running_quota():
    """C8: the fake/trace path only derives public outputs — no tuner
    consultation, no bench, no RUNNING contract consumed."""
    from cute_gemm.gemm_interface import gemm_tuned

    contracts_before = set(gemm_tuned._measured_contracts)
    cache_before = dict(gemm_tuned.cache)
    with FakeTensorMode():
        A = torch.empty(512, 512, dtype=torch.bfloat16, device="cuda")
        B = torch.empty(512, 512, dtype=torch.bfloat16, device="cuda")
        out = torch.ops.dlkernel.gemm(A, B, tune_mode=int(TuneMode.RUNNING))
        assert out.shape == (512, 512) and out.dtype == torch.bfloat16
    assert set(gemm_tuned._measured_contracts) == contracts_before
    assert dict(gemm_tuned.cache) == cache_before


def test_running_real_framework_lifecycle_first_tune_then_heuristic_then_disk_hit(
    tmp_path, monkeypatch
):
    """User-mandated case (2026-10-21): the REAL framework lifecycle under
    TuneMode.RUNNING on the actual varlen entry, with every artifact inside
    pytest's ``tmp_path`` (zero garbage left behind):

    1. cold call            -> one in-process sweep (RUNNING quota consumed)
    2. new winner key (same fixed contract F) -> the ``-1`` heuristic path:
       NO bench, exactly one RuntimeWarning, output still numerically correct
    3. the winner for that key exists on disk (published by another process)
       -> the next call loads it: NO bench, NO heuristic (a measured winner
       beats the fallback; misses are never memoized negatively)
    """
    import warnings

    import torch

    from cute_gemm.gemm_config import TuneMode
    from cute_gemm.gemm_interface import gemm, gemm_tuned

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gemm_tuned, "_measured_contracts", set())
    gemm_tuned.cache.clear()

    benches = {"n": 0}
    real_bench = gemm_tuned._bench

    def counting_bench(*a, **kw):
        benches["n"] += 1
        return real_bench(*a, **kw)

    monkeypatch.setattr(gemm_tuned, "_bench", counting_bench)

    def make(tokens):
        torch.manual_seed(0)
        a = torch.randn(tokens, 2048, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(32, 2048, 512, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(tokens, 512, dtype=torch.bfloat16, device="cuda")
        cu = torch.arange(33, dtype=torch.int32, device="cuda") * (tokens // 32)
        ref = torch.cat([a[cu[e] : cu[e + 1]].float() @ b[e].float() for e in range(32)])
        return a, b, out, cu, ref

    atol = rtol = 5e-2

    # Phase 1: cold RUNNING call tunes once and persists the winner (in tmp_path).
    a1, b1, out1, cu1, ref1 = make(4096)
    gemm(a1, b1, out1, cu_seqlens_m=cu1, tune_mode=TuneMode.RUNNING)
    assert benches["n"] > 0
    torch.testing.assert_close(out1.float(), ref1, atol=atol, rtol=rtol)
    n_after_phase1 = benches["n"]
    assert len(gemm_tuned._measured_contracts) == 1

    # Phase 2: token bucket moves -> new W, same contract F -> the 2026-10-22
    # reuse semantics: the contract's first online tune transfers across
    # buckets.  No bench, NO cliff warning (a measured config is in use), and
    # the fn runs with the F winner explicit (plan caching allowed).
    a2, b2, out2, cu2, ref2 = make(123)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gemm(a2, b2, out2, cu_seqlens_m=cu2, tune_mode=TuneMode.RUNNING)
    assert benches["n"] == n_after_phase1  # no online tuning
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert gemm_tuned.used_fallback is False  # measured winner, not the -1 path
    assert gemm_tuned.best_config is not None
    # 123 tokens: cu covers only 96 rows, and the kernel writes exactly those.
    torch.testing.assert_close(
        out2[: ref2.shape[0]].float(), ref2, atol=atol, rtol=rtol
    )  # transferred winner legal

    # Phase 3: another process publishes key2's winner (a TUNED sweep here
    # stands in for it); a FRESH process view (cleared in-memory state) then
    # hits the disk winner: no bench, no heuristic warning, measured result.
    gemm(a2, b2, out2, cu_seqlens_m=cu2, tune_mode=TuneMode.TUNED)
    n_publish = benches["n"]
    gemm_tuned.cache.clear()  # fresh-process view; disk state persists
    monkeypatch.setattr(gemm_tuned, "_fallback_warned", set())  # warnings audible again
    with warnings.catch_warnings(record=True) as caught3:
        warnings.simplefilter("always")
        gemm(a2, b2, out2, cu_seqlens_m=cu2, tune_mode=TuneMode.RUNNING)
    assert benches["n"] == n_publish  # disk hit: neither tune nor fallback
    assert not [w for w in caught3 if issubclass(w.category, RuntimeWarning)]
    torch.testing.assert_close(out2[: ref2.shape[0]].float(), ref2, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode_name", ["STATIC", "RUNNING"])
def test_eager_gemm_forwards_tune_mode_no_sweep_on_static(tmp_path, monkeypatch, mode_name):
    """Review F1 regression (2026-10-21): the five ``fn = gemm_tuned`` call
    sites dropped ``tune_mode`` for a whole day -- every real call silently
    ran as RUNNING.  STATIC's contract is ZERO benches on the very first
    (cold-cache) call through the public eager entry.  GPU test; the CPU
    twin of the forwarding chain is test_pinned_config_bypasses_tuning."""
    import torch

    from cute_gemm.gemm_config import TuneMode
    from cute_gemm.gemm_interface import gemm, gemm_tuned

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    gemm_tuned.cache.clear()
    monkeypatch.setattr(gemm_tuned, "_measured_contracts", set())

    benches = {"n": 0}
    real_bench = gemm_tuned._bench

    def counting_bench(*a, **kw):
        benches["n"] += 1
        return real_bench(*a, **kw)

    monkeypatch.setattr(gemm_tuned, "_bench", counting_bench)

    torch.manual_seed(0)
    a = torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(32, 2048, 512, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(4096, 512, dtype=torch.bfloat16, device="cuda")
    cu = torch.arange(33, dtype=torch.int32, device="cuda") * 128

    gemm(a, b, out, cu_seqlens_m=cu, tune_mode=getattr(TuneMode, mode_name))
    if mode_name == "STATIC":
        assert benches["n"] == 0  # never measures, first call included
    else:
        assert benches["n"] > 0  # RUNNING's first cold contract tunes once
