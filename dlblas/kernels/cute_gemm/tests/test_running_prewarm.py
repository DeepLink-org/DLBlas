"""RUNNING prepares every token bucket before returning its first real call."""

from contextlib import nullcontext

import pytest
import torch

from cute_gemm.autotuner import AutotuneConfig, Autotuner, TunePolicy
from cute_gemm.gemm_config import TuneMode, get_all_configs


@pytest.fixture(autouse=True)
def cpu_target(monkeypatch, tmp_path):
    import cute_gemm.autotuner as at
    from cute_gemm.cache import async_compile

    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    monkeypatch.setenv("DLKERNEL_SM_COUNT", "132")
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DLKERNEL_CACHE_AUTOTUNING", raising=False)
    monkeypatch.setattr(at, "_gpu_warmup", lambda: None)
    monkeypatch.setattr(async_compile, "pool_scope", lambda: nullcontext(None))


def _harness(monkeypatch):
    import importlib

    from cute_gemm.autotuner import RunningConfig

    precompile = importlib.import_module("cute_gemm.cache.precompile")
    events = []

    class Policy(TunePolicy):
        def prepares_running(self, key):
            return True

        def make_key(self, args, kwargs, default_key):
            return (args[0].numel(),)

        def make_contract(self, args, kwargs, default_key):
            return ("fixed",)

        def running_configs(self, args, kwargs, configs):
            events.append("select")
            return [
                RunningConfig((n,), {"x": torch.empty(n, device="meta")}, configs[1], configs)
                for n in (2, 4, 8)
            ]

    def fn(x, config=None, tune_mode=None):
        if x.device.type == "meta":
            events.append(("compile", x.numel(), config))
            return
        events.append(("execute", x.numel(), config))
        return x * config

    def bench(call, **kw):
        events.append("bench")
        call()
        return [1.0, 1.0, 1.0]

    def compile_calls(calls):
        for call in calls:
            call()

    monkeypatch.setattr(precompile, "precompile", compile_calls)
    tuner = Autotuner(
        fn,
        [],
        [AutotuneConfig(config=1), AutotuneConfig(config=2)],
        policy=Policy(),
        do_bench=bench,
    )
    return tuner, events


def test_running_prepares_all_buckets_then_only_uses_cache(monkeypatch):
    tuner, events = _harness(monkeypatch)
    torch.testing.assert_close(tuner(torch.ones(2), tune_mode=TuneMode.RUNNING), torch.ones(2))
    assert events.count("bench") == 2
    assert {e[1] for e in events if isinstance(e, tuple) and e[0] == "compile"} == {2, 4, 8}
    assert events[-1] == ("execute", 2, 1)
    assert set(tuner.cache) == {(2,)}  # heuristic choices are not measured winners
    events.clear()
    for n in (8, 4, 2):
        expected = torch.ones(n) * (1 if n == 2 else 2)
        torch.testing.assert_close(tuner(torch.ones(n), tune_mode=TuneMode.RUNNING), expected)
    assert events == [("execute", 8, 2), ("execute", 4, 2), ("execute", 2, 1)]


def test_running_memory_winner_still_prepares_and_does_not_rebench(monkeypatch):
    tuner, events = _harness(monkeypatch)
    tuner.cache[(2,)] = tuner.configs[0]
    tuner(torch.ones(2), tune_mode=TuneMode.RUNNING)
    assert "bench" not in events
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "compile"]) == 3
    events.clear()
    torch.testing.assert_close(
        tuner(torch.ones(8), tune_mode=TuneMode.RUNNING), torch.full((8,), 2.0)
    )
    assert events == [("execute", 8, 2)]


def test_failed_prewarm_is_not_published_or_rebenchmarked(monkeypatch):
    import importlib

    tuner, events = _harness(monkeypatch)
    module = importlib.import_module("cute_gemm.cache.precompile")
    compile_calls = module.precompile

    def fail(calls):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(module, "precompile", fail)
    with pytest.raises(RuntimeError, match="compile failed"):
        tuner(torch.ones(2), tune_mode=TuneMode.RUNNING)
    assert not tuner._running_cache
    assert not tuner._running_ready
    monkeypatch.setattr(module, "precompile", compile_calls)
    tuner(torch.ones(2), tune_mode=TuneMode.RUNNING)
    assert events.count("bench") == 2


def test_tuned_does_not_accept_running_heuristics_as_measured(monkeypatch):
    tuner, events = _harness(monkeypatch)
    tuner(torch.ones(2), tune_mode=TuneMode.RUNNING)
    events.clear()
    torch.testing.assert_close(tuner(torch.ones(8), tune_mode=TuneMode.TUNED), torch.ones(8))
    assert events.count("bench") == 2


def _varlen_call(rows, direction, *, experts=64, n=2048, k=1024):
    cu = torch.empty(experts + 1, dtype=torch.int32, device="meta")
    if direction == "m":
        return dict(
            A=torch.empty(rows, k, dtype=torch.bfloat16, device="meta"),
            B=torch.empty(experts, k, n, dtype=torch.bfloat16, device="meta"),
            out=torch.empty(rows, n, dtype=torch.bfloat16, device="meta"),
            cu_seqlens_m=cu,
            cu_seqlens_k=None,
        )
    return dict(
        A=torch.empty(rows, k, dtype=torch.bfloat16, device="meta").T,
        B=torch.empty(rows, n, dtype=torch.bfloat16, device="meta"),
        out=torch.empty(experts, k, n, dtype=torch.bfloat16, device="meta"),
        cu_seqlens_m=None,
        cu_seqlens_k=cu,
    )


@pytest.mark.parametrize("direction,expected", [("m", 8), ("k", 10)])
def test_varlen_prewarm_covers_seven_average_buckets_and_leaf_splits(direction, expected):
    from cute_gemm.varlen_selector import VarlenGemmTunePolicy

    policy = VarlenGemmTunePolicy()
    call = _varlen_call(4096, direction)
    policy.bind_signature(tuple(call), (None,) * len(call))
    configs = [AutotuneConfig(config=c) for c in get_all_configs()]
    cases = policy.running_configs((), call, configs)
    by_key = {case.key: case for case in cases}
    assert len(by_key) == expected
    # Bucket boundaries, selector cuts, envelope edges, and the reported 2**20 stall.
    points = {1, 7, 8, 4098424, 4098425, 2**20, 2**31 - 1, 76458, 76459, 161940, 161941}
    for exponent in range(5, 11):
        points.update((64 * 2**exponent - 1, 64 * 2**exponent, 64 * 2**exponent + 1))
    for rows in sorted(p for p in points if p < 2**31):
        actual = _varlen_call(rows, direction)
        key = policy.make_key((), actual, lambda: None)
        assert key in by_key, rows
        prepared = policy.prepare_candidates(configs, actual, {})
        assert by_key[key].config in prepared


def test_epilogue_prewarm_resizes_declared_tensor_roles():
    from cute_gemm.epilogue.library import dgated_mod
    from cute_gemm.gemm_epilogue_tune_policy import GemmEpilogueTunePolicy

    mod = dgated_mod("swiglu", has_scale=True, has_reduce=True)
    policy = GemmEpilogueTunePolicy(mod)
    call = _varlen_call(4096, "m")
    call["D"] = call.pop("out")
    call["B"] = call["B"].mT  # epilogue convention (E, N, K)
    call["C"] = torch.empty(4096, 4096, dtype=torch.bfloat16, device="meta")
    call["mAuxOut"] = torch.empty(4096, 2048, dtype=torch.bfloat16, device="meta")
    call["mColVecBroadcast"] = torch.empty(4096, device="meta")
    call["mColVecReduce"] = torch.empty(4096, device="meta")
    policy.bind_signature(tuple(call), (None,) * len(call))
    configs = [AutotuneConfig(config=c) for c in get_all_configs()]
    cases = policy.running_configs((), call, configs)
    for case in cases:
        rows = case.kwargs["A"].shape[0]
        assert case.kwargs["C"].shape == (rows, 4096)
        assert case.kwargs["mAuxOut"].shape == (rows, 2048)
        assert case.kwargs["mColVecBroadcast"].shape == (rows,)
        assert policy.make_contract((), case.kwargs, lambda: None) == policy.make_contract(
            (), call, lambda: None
        )


def test_average_buckets_have_one_open_tail():
    from cute_gemm.varlen_selector import average_rows_bucket

    pairs = [
        (1, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 128),
        (129, 256),
        (257, 512),
        (513, 1024),
        (1024, 1024),
        (1025, 2048),
        (8192, 2048),
        (2**24, 2048),
    ]
    for average, expected in pairs:
        assert average_rows_bucket(average * 64, 64, device_capability=(9, 0)) == expected
    assert average_rows_bucket(1024 * 64 + 1, 64, device_capability=(9, 0)) == 2048


@pytest.mark.parametrize("arch", ["100", "120"])
def test_other_architectures_keep_their_existing_bucket_identity(monkeypatch, arch):
    import cute_gemm.varlen_selector as selector
    from cute_gemm.varlen_selector import VarlenGemmTunePolicy, average_rows_bucket

    capability = (int(arch) // 10, int(arch) % 10)
    monkeypatch.setattr(selector, "get_device_capacity", lambda device: capability)
    call = _varlen_call(64 * 8192, "m")
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(tuple(call), (None,) * len(call))
    key = policy.make_key((), call, lambda: None)
    assert key[0] == "gemm_token_bucket_v5"
    assert ("avg_expert_m", 8192) in key
    assert average_rows_bucket(64 * 8192, 64) == 8192
    assert not policy.prepares_running(key)
    assert average_rows_bucket(64 * 8192, 64, device_capability=capability) == 8192


def test_meta_preparation_uses_the_real_calls_device_identity(monkeypatch):
    import cute_gemm.varlen_selector as selector

    policy = selector.VarlenGemmTunePolicy()
    call = dict(
        A=torch.empty(64, 64, dtype=torch.bfloat16),
        B=torch.empty(2, 64, 64, dtype=torch.bfloat16),
        out=torch.empty(64, 64, dtype=torch.bfloat16),
        cu_seqlens_m=torch.empty(3, dtype=torch.int32),
        cu_seqlens_k=None,
    )
    policy.bind_signature(tuple(call), (None,) * len(call))
    # A real CUDA call has an arch; a meta tensor without an override has
    # none. Model that distinction without constructing a CUDA context.
    monkeypatch.setattr(
        selector, "_device_capability", lambda t: (9, 0) if t.device.type == "cpu" else None
    )
    configs = [AutotuneConfig(config=c) for c in get_all_configs()]
    cases = policy.running_configs((), call, configs)
    assert policy.make_key((), call, lambda: None) in {case.key for case in cases}


@pytest.mark.parametrize("direction", ["m", "k"])
def test_prewarm_and_live_shapes_capture_identical_compile_keys(monkeypatch, direction):
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles
    from cute_gemm.gemm_interface import gemm_tuned

    call = _varlen_call(4096, direction)
    policy = gemm_tuned.policy
    cases = policy.running_configs((), call, gemm_tuned.configs)

    def capture(kwargs, config):
        with capture_compiles() as captured, pytest.raises(CompileCaptured):
            gemm_tuned.fn(**kwargs, **config.all_kwargs())
        return {s.sha for s in captured.tasks()}

    by_key = {c.key: c for c in cases}
    warmed = set().union(*(capture(c.kwargs, c.config) for c in cases))
    for rows in (7, 33 * 64, 513 * 64, 1025 * 64, 2**20, 4098425):
        actual = _varlen_call(rows, direction)
        selected = by_key[policy.make_key((), actual, lambda: None)].config
        assert capture(actual, selected) <= warmed


@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.float16, torch.float8_e4m3fn])
def test_dgated_prewarm_captures_real_epilogue_compile_boundaries(gather, score_dtype):
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles
    from cute_gemm.epilogue.library import dgated_mod
    from cute_gemm.fused_grouped_gemm.gemm_sm90 import GemmMoeSm90
    from cute_gemm.gemm_runtime.autotune import _get_tuner, sink_arg_shapes

    mod = GemmMoeSm90.bind_epilogue(dgated_mod("swiglu", has_scale=True, has_reduce=True))

    def workload(rows):
        call = _varlen_call(rows, "m")
        call["D"] = torch.empty(rows, 4096, device="meta", dtype=torch.bfloat16)
        call.pop("out")
        call["B"] = call["B"].mT
        call["C"] = torch.empty_like(call["D"])
        call["mAuxOut"] = torch.empty(rows, 2048, device="meta", dtype=torch.bfloat16)
        call["mColVecBroadcast"] = torch.empty(rows, device="meta", dtype=score_dtype)
        shapes = sink_arg_shapes(mod, rows, 2048, device="meta", num_seqs=64)
        call["mColVecReduce"] = torch.empty(shapes["mColVecReduce"], device="meta")
        call["b_kn"] = False
        call["dynamic_scheduler"] = True
        call["mod_digest"] = mod.semantic_digest
        if gather:
            call["A"] = torch.empty(4096, 1024, device="meta", dtype=torch.bfloat16)
            call["A_idx"] = torch.empty(rows, device="meta", dtype=torch.int32)
        return call

    call = workload(4096)
    tuner = _get_tuner(
        mod, ("mAuxOut", "mColVecBroadcast", "mColVecReduce"), True, call["A"].device
    )
    cases = tuner.policy.running_configs((), call, tuner.configs)
    with capture_compiles() as captured:
        for case in cases:
            with pytest.raises(CompileCaptured):
                tuner.fn(**(case.kwargs | case.config.all_kwargs()))
    warmed = {s.sha for s in captured.tasks()}
    by_key = {c.key: c for c in cases}
    for rows in (1, 7, 2049, 65537, 76459, 2**20):
        actual = workload(rows)
        config = by_key[tuner.policy.make_key((), actual, lambda: None)].config
        with capture_compiles() as live, pytest.raises(CompileCaptured):
            tuner.fn(**(actual | config.all_kwargs()))
        assert {s.sha for s in live.tasks()} <= warmed
