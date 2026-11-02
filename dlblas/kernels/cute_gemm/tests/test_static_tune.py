# Copyright (c) 2026, Tri Dao.
"""CPU-only tests for the offline tuner CLI's case machinery (static_tune).

The meta builders must hand each family's operator exactly its documented
shapes/dtypes; the prefill call factory must resolve operator-owned families
(fused_up/fused_down/epi) through their own STATIC chain on a GPU-blind box
(``DLKERNEL_ARCH`` set), reaching the jit compile boundary under capture --
never a sweep, never a launch.  No test touches a CUDA device.
"""

import itertools
import json
import os

import pytest
import torch

from cute_gemm import static_tune as st


@pytest.fixture(autouse=True)
def _gpu_blind(monkeypatch, tmp_path):
    """GPU-blind box: capability answers come from the arch override and
    winner reads/writes stay inside pytest's tmp_path."""
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path / "tune"))


UP_CASE = {"family": "fused_up", "tokens": 32, "experts": 8, "k": 64, "n": 64, "dtype": "bfloat16"}
DOWN_CASE = {
    "family": "fused_down",
    "tokens": 32,
    "experts": 8,
    "k": 32,
    "n": 64,
    "dtype": "bfloat16",
}
EPI_CASE = {"family": "epi", "m": 128, "k": 64, "n": 96, "dtype": "bfloat16"}


def test_meta_builder_fused_up_matches_op_contract():
    """moe_up_proj_act: x [R, H], gate_up_weight [E, 2I, H] (swiglu concat);
    the op allocates (postact, preact) itself, so the out slot stays None."""
    (x, w13, out, kw) = st._run_case_tensors_meta(UP_CASE)
    assert x.shape == (32, 64) and x.dtype is torch.bfloat16
    assert w13.shape == (8, 64, 64) and w13.dtype is torch.bfloat16
    assert w13.shape[1] % 2 == 0  # the op rejects an odd 2I width
    assert x.is_meta and w13.is_meta
    assert out is None
    off = kw["expert_offsets"]
    assert off.shape == (8 + 1,) and off.dtype is torch.int32 and off.is_meta


def test_meta_builder_fused_down_matches_op_contract():
    """moe_down_proj_dact: grad_output [R, H], down_weight [E, H, I]; the
    swiglu preact [R, 2I] is an INPUT riding the tuple's third slot."""
    (go, w2, preact, kw) = st._run_case_tensors_meta(DOWN_CASE)
    assert go.shape == (32, 64) and go.dtype is torch.bfloat16
    assert w2.shape == (8, 64, 32) and w2.dtype is torch.bfloat16
    assert preact.shape == (32, 64) and preact.dtype is torch.bfloat16
    assert preact.shape[1] == 2 * w2.shape[2]  # gated swiglu preact width
    assert go.is_meta and w2.is_meta and preact.is_meta
    off = kw["expert_offsets"]
    assert off.shape == (9,) and off.dtype is torch.int32 and off.is_meta


def test_meta_builder_epi_matches_op_contract():
    """_epi_bias_double: A [m, k], B [k, n], fp32 [1, n] row bias; D is
    allocated by the op (out=None)."""
    (a, b, out, kw) = st._run_case_tensors_meta(EPI_CASE)
    assert a.shape == (128, 64) and a.dtype is torch.bfloat16
    assert b.shape == (64, 96) and b.dtype is torch.bfloat16
    assert a.is_meta and b.is_meta
    assert out is None
    bias = kw["bias"]
    assert bias.shape == (1, 96) and bias.dtype is torch.float32 and bias.is_meta


def test_expert_offsets_contract():
    """The fused families' routed-row contract: expert-major cumulative
    offsets, last entry == the routed row total."""
    off = st._expert_offsets(8, 32, "cpu")
    assert off.dtype is torch.int32
    assert off.tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 32]


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("tokens,experts", [(1, 8), (123, 32), (32768, 32), (96, 32)])
def test_expert_offsets_preserve_total(tokens, experts, device):
    """The (i*T)//E prefix is monotone, starts at 0 and ends EXACTLY at T;
    per-expert counts differ by at most one.  The old arange*(T//E) form
    dropped the remainder (T=123/E=32 covered 96 rows; T<E covered none)."""
    off = st._expert_offsets(experts, tokens, device)
    assert off.dtype is torch.int32 and off.device.type == device
    assert off.shape == (experts + 1,)
    if device == "meta":
        return  # meta tensors carry no data; the cpu leg pins the values
    bounds = off.tolist()
    assert len(bounds) == experts + 1
    assert bounds[0] == 0 and bounds[-1] == tokens
    deltas = [hi - lo for lo, hi in itertools.pairwise(bounds)]
    assert all(d >= 0 for d in deltas)
    assert max(deltas) - min(deltas) <= 1
    assert set(deltas) <= {tokens // experts, -(-tokens // experts)}


class _FakeTuner:
    """Stand-in for an Autotuner in the epi seam's registry: the two
    in-memory state holders ``_reset_inprocess_tuners`` must clear."""

    def __init__(self):
        self.cache = {}
        self._measured_contracts = set()


def test_study_run_case_resets_inprocess_tuner_state(tmp_path, monkeypatch):
    """Two same-bucket raw cases in study mode must not share in-process
    tuner state: at the start of each study case the winner dicts are empty,
    so the case re-measures (the stub re-dirties them per launch).  Production
    deliberately shares winners across calls — the reset stays study-only."""
    from cute_gemm.gemm_interface import gemm_tuned
    from cute_gemm.gemm_runtime import autotune as epi_autotune

    fake_tuner = _FakeTuner()
    monkeypatch.setitem(epi_autotune._MOD_TUNERS, ("fake-mod",), fake_tuner)
    monkeypatch.setattr(gemm_tuned, "cache", {})
    monkeypatch.setattr(gemm_tuned, "_measured_contracts", set())

    def snapshot():
        return (
            dict(gemm_tuned.cache),
            set(gemm_tuned._measured_contracts),
            dict(fake_tuner.cache),
            set(fake_tuner._measured_contracts),
        )

    def sweep():
        # What a real TUNED call leaves in memory after measuring; the same
        # bucket key for both cases (e.g. tokens 32767 vs 32768, same E).
        gemm_tuned.cache[("bucket",)] = "winner"
        gemm_tuned._measured_contracts.add("contract")
        fake_tuner.cache[("bucket",)] = "winner"
        fake_tuner._measured_contracts.add("contract")

    case_a = dict(EPI_CASE, seed=1)
    case_b = dict(EPI_CASE, seed=2)
    (a, b, out, kw) = st._run_case_tensors_meta(case_a)  # real builder needs CUDA
    monkeypatch.setattr(st, "_run_case_tensors", lambda c: (a, b, out, kw))
    seen = []
    monkeypatch.setattr(
        st,
        "_launch_case_op",
        lambda c, ta, tb, to, tkw, tm: seen.append(snapshot()) or sweep(),
    )

    st._run_case(case_a, "study", tmp_path / "winners", tmp_path)
    st._run_case(case_b, "study", tmp_path / "winners", tmp_path)
    # The second study case started clean: it re-measures instead of
    # replaying case A's in-memory winner (fails under the old leak).
    assert seen[0] == ({}, set(), {}, set())
    assert seen[1] == ({}, set(), {}, set())

    # Fresh process for the production leg (the study sweeps left dirt).
    gemm_tuned.cache.clear()
    gemm_tuned._measured_contracts.clear()
    fake_tuner.cache.clear()
    fake_tuner._measured_contracts.clear()
    st._run_case(case_a, "production", tmp_path / "winners", tmp_path)
    st._run_case(case_b, "production", tmp_path / "winners", tmp_path)
    assert seen[2] == ({}, set(), {}, set())
    assert seen[3] == ({("bucket",): "winner"}, {"contract"}, {("bucket",): "winner"}, {"contract"})


def test_iter_cases_enumerates_new_families_with_existing(tmp_path):
    """Manifest order is preserved across the mixed families and sharding is
    a plain slice over it -- new families need no special casing."""
    cases = [
        {"family": "dense", "m": 128, "k": 64, "n": 64},
        UP_CASE,
        {"family": "varlen_m", "tokens": 32, "experts": 8, "k": 64, "n": 64},
        DOWN_CASE,
        {"family": "varlen_k", "tokens": 32, "experts": 8, "k": 64, "n": 64, "m": 128},
        EPI_CASE,
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(cases))
    assert [c["family"] for c in st._iter_cases(manifest, None)] == [
        "dense",
        "fused_up",
        "varlen_m",
        "fused_down",
        "varlen_k",
        "epi",
    ]
    assert [c["family"] for c in st._iter_cases(manifest, "1/2")] == [
        "fused_up",
        "fused_down",
        "epi",
    ]


def test_case_slug_stable_for_new_families():
    """The slug is a pure function of the case and renders on the fixed key
    set (family, m, n, k, experts, tokens, dtype, seed) with no per-family
    special casing."""
    assert st._case_slug(UP_CASE) == "familyfused_up_n64_k64_experts8_tokens32_dtypebfloat16"
    assert st._case_slug(DOWN_CASE) == "familyfused_down_n64_k32_experts8_tokens32_dtypebfloat16"
    assert st._case_slug(EPI_CASE) == "familyepi_m128_n96_k64_dtypebfloat16"
    assert st._case_slug(dict(UP_CASE)) == st._case_slug(UP_CASE)
    assert st._case_slug(dict(UP_CASE, seed=7)) == (
        "familyfused_up_n64_k64_experts8_tokens32_dtypebfloat16_seed7"
    )


def test_case_slug_distinguishes_new_family_dims():
    """Dims ride the slug's fixed keys, so two cases differing only in a
    contraction/output dim never share a winner dir."""
    assert st._case_slug(dict(UP_CASE, k=128)) != st._case_slug(UP_CASE)
    assert st._case_slug(dict(DOWN_CASE, n=96)) != st._case_slug(DOWN_CASE)
    assert st._case_slug(dict(EPI_CASE, n=192)) != st._case_slug(EPI_CASE)


def test_launch_dispatch_threads_tune_mode_and_slots(monkeypatch):
    """Each operator-owned family goes to its own public entry with the
    tuple's slots in the op's positional order; the GEMM families keep
    ``gemm(a, b, out, **kw)``.  study/production thread TUNED through it."""
    import cute_gemm.fused_grouped_gemm.interface as moe
    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import TuneMode

    seen = {}

    def up(a, b, off, *, tune_mode):
        seen["fused_up"] = (a, b, off, tune_mode)

    def down(a, b, preact, off, *, tune_mode):
        seen["fused_down"] = (a, b, preact, off, tune_mode)

    def epi(a, b, *, bias, tune_mode):
        seen["epi"] = (a, b, bias, tune_mode)

    def gemm(a, b, out, *, tune_mode):
        seen["dense"] = (a, b, out, tune_mode)

    monkeypatch.setattr(moe, "moe_up_proj_act", up)
    monkeypatch.setattr(moe, "moe_down_proj_dact", down)
    monkeypatch.setattr(st, "_epi_bias_double", epi)
    monkeypatch.setattr(gi, "gemm", gemm)

    (x, w13, _, kw_up) = st._run_case_tensors_meta(UP_CASE)
    st._launch_case_op(UP_CASE, x, w13, None, kw_up, TuneMode.TUNED)
    assert seen["fused_up"] == (x, w13, kw_up["expert_offsets"], TuneMode.TUNED)

    (go, w2, preact, kw_down) = st._run_case_tensors_meta(DOWN_CASE)
    st._launch_case_op(DOWN_CASE, go, w2, preact, kw_down, TuneMode.TUNED)
    assert seen["fused_down"] == (go, w2, preact, kw_down["expert_offsets"], TuneMode.TUNED)

    (a, b, _, kw_epi) = st._run_case_tensors_meta(EPI_CASE)
    st._launch_case_op(EPI_CASE, a, b, None, kw_epi, TuneMode.TUNED)
    assert seen["epi"] == (a, b, kw_epi["bias"], TuneMode.TUNED)

    dense = {"family": "dense", "m": 128, "k": 64, "n": 64}
    st._launch_case_op(dense, a, b, "out-buffer", {}, TuneMode.TUNED)
    assert seen["dense"] == (a, b, "out-buffer", TuneMode.TUNED)


def test_run_case_pins_tuned_and_isolates_winner_dir(tmp_path, monkeypatch):
    """study drives the dispatcher with TUNED under a per-case winner dir:
    the env swap takes effect and the study dataset stays empty (nothing
    swept here -- the real tensors/ops are stubbed at the same boundary)."""
    from cute_gemm.gemm_config import TuneMode

    case = dict(EPI_CASE, seed=3)
    (a, b, out, kw) = st._run_case_tensors_meta(case)  # real builder needs CUDA
    seen = {}
    monkeypatch.setattr(st, "_run_case_tensors", lambda c: (a, b, out, kw))
    monkeypatch.setattr(
        st,
        "_launch_case_op",
        lambda c, ta, tb, to, tkw, tm: seen.update(case=c, tune_mode=tm, tensors=(ta, tb, to, tkw)),
    )
    winners = tmp_path / "winners"
    record = st._run_case(case, "study", winners, tmp_path)

    assert seen["case"] is case or seen["case"] == case
    assert seen["tune_mode"] is TuneMode.TUNED
    assert seen["tensors"] == (a, b, out, kw)
    expected_dir = winners / st._case_slug(case)
    assert record["winner_dir"] == str(expected_dir)
    assert expected_dir.is_dir()
    assert os.environ["DLKERNEL_CACHE_DIR"] == str(expected_dir)
    dataset = tmp_path / "study_dataset.jsonl"
    assert record["dataset_path"] == str(dataset)
    assert dataset.read_text() == ""  # no fabricated timings: nothing swept here


@pytest.mark.parametrize(
    "case",
    [UP_CASE, DOWN_CASE, EPI_CASE],
    ids=["fused_up", "fused_down", "epi"],
)
def test_prefill_call_op_owned_resolves_static_gpu_blind(case):
    """prefill's operator-owned branch reaches the real compile boundary on a
    GPU-blind box through the operator's own STATIC chain: the analytical-
    fallback warning proves STATIC resolved the config (a TUNED call would
    have swept, a pinned one never consults the tuner), and exactly one
    boundary is captured (single-kernel scope aborts the call)."""
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles

    call = st._prefill_call(case)
    with (
        capture_compiles() as captured,
        pytest.warns(RuntimeWarning, match="analytical fallback"),
        pytest.raises(CompileCaptured),
    ):
        call()
    tasks = captured.tasks()
    assert len(tasks) == 1
    assert tasks[0].fn.__qualname__ == "_compile_gemm_epi"
