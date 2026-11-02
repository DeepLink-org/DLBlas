"""Real ordinary/fused calls must not tune or compile after RUNNING startup."""

import importlib

import pytest
import torch

from cute_gemm import TuneMode, moe_down_proj_dact, moe_up_proj_act
from cute_gemm.gemm_interface import gemm, gemm_tuned

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="RUNNING analytical prewarm targets SM90",
)


@requires_sm90
@pytest.mark.parametrize("family", ["m", "k", "fused_up", "fused_dact", "fused_dgated"])
def test_running_first_call_prepares_later_buckets(family, tmp_path, monkeypatch):
    from test_fused_grouped_gemm_sm90 import _assert_close_scaled, _down_reference, _up_reference

    import cute_gemm.gemm_runtime.autotune as epi_tune
    from cute_gemm.autotuner import Autotuner
    from cute_gemm.gemm import _compile_gemm
    from cute_gemm.gemm_runtime.host import _compile_gemm_epi

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path / "winners"))
    monkeypatch.setenv("DLKERNEL_ARTIFACT_CACHE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "10")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "16")
    monkeypatch.delenv("DLKERNEL_REQUIRE_TUNE_CACHE", raising=False)
    monkeypatch.delenv("DLKERNEL_FORCE_CACHE_UPDATE", raising=False)
    monkeypatch.setattr(epi_tune, "_MOD_TUNERS", {})
    for attr, value in (
        ("cache", {}),
        ("_running_cache", {}),
        ("_running_ready", set()),
        ("_measured_contracts", set()),
        ("_contract_winners", {}),
        ("_reuse_resolved", {}),
    ):
        monkeypatch.setattr(gemm_tuned, attr, value)
    sweeps = []
    check_disk_cache = Autotuner.check_disk_cache

    def count_sweep(self, key, configs, benchmark):
        def measured():
            sweeps.append(self.fn.__name__)
            return benchmark()

        return check_disk_cache(self, key, configs, measured)

    monkeypatch.setattr(Autotuner, "check_disk_cache", count_sweep)
    torch.manual_seed(2026)
    experts, width, n = 32, 2048, 512
    weights = torch.randn(experts, width, n, device="cuda", dtype=torch.bfloat16) / 32
    up_weights = (
        torch.randn(experts, 2 * n, width, device="cuda", dtype=torch.bfloat16) / 32
        if family == "fused_up"
        else None
    )

    def run(rows):
        # Preserve the total, including non-divisible and empty-expert cases.
        offsets = (
            torch.arange(experts + 1, device="cuda", dtype=torch.int64) * rows // experts
        ).int()
        if family == "k":
            a = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16) / 16
            got = gemm(a.T, b, cu_seqlens_k=offsets, tune_mode=TuneMode.RUNNING)
            expected = torch.stack(
                [
                    a[offsets[e] : offsets[e + 1]].float().T
                    @ b[offsets[e] : offsets[e + 1]].float()
                    for e in range(experts)
                ]
            )
            _assert_close_scaled(got, expected, family, torch.bfloat16)
        else:
            a = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) / 4
            if family == "m":
                got = gemm(a, weights, cu_seqlens_m=offsets, tune_mode=TuneMode.RUNNING)
                expected = torch.cat(
                    [
                        a[offsets[e] : offsets[e + 1]].float() @ weights[e].float()
                        for e in range(experts)
                    ]
                )
                _assert_close_scaled(got, expected, family, torch.bfloat16)
            elif family == "fused_up":
                postact, preact = moe_up_proj_act(
                    a, up_weights, offsets, tune_mode=TuneMode.RUNNING
                )
                ref_preact, ref_postact = _up_reference(
                    a, up_weights, None, offsets, torch.arange(rows, device="cuda"), "swiglu"
                )
                _assert_close_scaled(preact, ref_preact, "preact", preact.dtype)
                _assert_close_scaled(postact, ref_postact, "postact", postact.dtype)
            else:
                activation = "silu" if family == "fused_dact" else "swiglu"
                preact = torch.randn(
                    rows, n if activation == "silu" else 2 * n, device="cuda", dtype=torch.bfloat16
                )
                scores = torch.rand(rows, device="cuda")
                got = moe_down_proj_dact(
                    a,
                    weights,
                    preact,
                    offsets,
                    router_weights=scores,
                    compute_router_weight_grad=True,
                    activation=activation,
                    tune_mode=TuneMode.RUNNING,
                )
                expected = _down_reference(
                    a,
                    weights,
                    preact,
                    offsets,
                    torch.arange(rows, device="cuda"),
                    scores,
                    activation,
                )
                for name, output, reference in zip(("dH", "postact", "dS"), got, expected):
                    _assert_close_scaled(output, reference, name, output.dtype)
        torch.cuda.synchronize()

    run(experts * 16)
    assert len(sweeps) == 1, "A fused operator must not enter two online tuners"
    misses = (_compile_gemm.cache_info().misses, _compile_gemm_epi.cache_info().misses)
    winner_files = {p: p.stat().st_mtime_ns for p in (tmp_path / "winners").rglob("*.json")}

    def forbidden(*args, **kwargs):
        pytest.fail("a later RUNNING bucket attempted selection, compilation preparation, or bench")

    monkeypatch.setattr(Autotuner, "_bench", forbidden)
    monkeypatch.setattr(
        importlib.import_module("cute_gemm.gemm_running"), "running_configs", forbidden
    )
    monkeypatch.setattr(importlib.import_module("cute_gemm.gemm_heuristic"), "rank", forbidden)
    monkeypatch.setattr(
        importlib.import_module("cute_gemm.cache.precompile"), "precompile", forbidden
    )
    for rows in (
        7,
        experts * 33,
        experts * 65,
        experts * 129,
        experts * 257,
        experts * 513,
        experts * 1025,
        experts * 1200,
        2**20,
    ):
        run(rows)
        assert (_compile_gemm.cache_info().misses, _compile_gemm_epi.cache_info().misses) == misses
    assert len(sweeps) == 1
    assert {p: p.stat().st_mtime_ns for p in (tmp_path / "winners").rglob("*.json")} == winner_files
