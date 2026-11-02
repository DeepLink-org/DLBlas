import inspect

import pytest

from cute_gemm.autotuner import AutotuneConfig, TunePolicy


@pytest.fixture(autouse=True)
def _cpu_arch_for_metadata_key(monkeypatch):
    """Give metadata-only key tests a deterministic SM90 capability on CPU."""
    monkeypatch.setenv("DLKERNEL_ARCH", "90")


def _select_via_policy(base, call):
    """The tuner's prepare -> shortlist composition for varlen candidate tests."""
    from cute_gemm.varlen_selector import VarlenGemmTunePolicy

    policy = VarlenGemmTunePolicy()
    prepared = policy.prepare_candidates(base, call, {})
    return policy.shortlist(prepared, call, {})


def _bucket_key(args, kwargs=None):
    """Varlen tune key, with the ``gemm_tuned`` signature supplied for tests.

    The key is built from positional arguments, so the policy needs to know the
    parameter names; in production ``Autotuner`` passes its own ``arg_names``.
    """
    from cute_gemm.gemm_interface import gemm_tuned
    from cute_gemm.varlen_selector import gemm_tune_key

    return gemm_tune_key(args, kwargs or {}, gemm_tuned.arg_names)


def _contract_key(args, kwargs=None):
    """Fixed-contract identity (F), mirroring ``_bucket_key``'s W harness."""
    from cute_gemm.gemm_interface import gemm_tuned
    from cute_gemm.varlen_selector import gemm_contract_key

    return gemm_contract_key(args, kwargs or {}, gemm_tuned.arg_names)


def _varlen_m_args(rows, *, experts=32, k=2048, n=1024, dtype=None, b=None, cu=None):
    """Positional ``gemm_tuned`` args for a varlen-M call with packed expert-major B."""
    import torch

    dtype = dtype or torch.bfloat16
    b = torch.empty((experts, k, n), dtype=dtype) if b is None else b
    cu = torch.zeros((experts + 1,), dtype=torch.int32) if cu is None else cu
    return (
        torch.empty((rows, k), dtype=dtype),
        b,
        torch.empty((rows, n), dtype=dtype),
        None,
        None,
        1.0,
        1.0,
        cu,
        None,
        None,
    )


def test_gemm_contract_key_strips_declared_token_dims_only():
    # C3: only the declared token totals differ (same E/K/N weights geometry).
    assert _contract_key(_varlen_m_args(32768)) == _contract_key(_varlen_m_args(60000))
    # W still splits the same pair across buckets.
    assert _bucket_key(_varlen_m_args(32768)) != _bucket_key(_varlen_m_args(60000))


def test_gemm_contract_key_separates_fixed_geometry():
    import torch

    base = _contract_key(_varlen_m_args(4096))
    # C1: different expert geometry (Qwen vs DeepSeek style E) is a new contract.
    other_e = _contract_key(
        _varlen_m_args(
            4096,
            experts=16,
            b=torch.empty((16, 2048, 1024), dtype=torch.bfloat16),
            cu=torch.zeros((17,), dtype=torch.int32),
        )
    )
    assert base != other_e
    # C2: projection width, dtype, and weight layout each form a new contract.
    assert base != _contract_key(_varlen_m_args(4096, n=2048))
    assert base != _contract_key(_varlen_m_args(4096, dtype=torch.float16))
    b_n_major = torch.empty((32, 1024, 2048), dtype=torch.bfloat16)  # (E, N, K)
    assert base != _contract_key(_varlen_m_args(4096, b=b_n_major.mT))
    # Same contract, only data address/values differ -> same F.
    assert base == _contract_key(_varlen_m_args(4096))


def test_gemm_contract_key_varlen_k_folds_only_contraction_total():
    import torch

    def k_args(total_k, experts=32, m=128, n=1024):
        return (
            torch.empty((m, total_k)),
            torch.empty((total_k, n)),
            torch.empty((experts, m, n)),
            None,
            None,
            1.0,
            1.0,
            None,
            torch.zeros((experts + 1,), dtype=torch.int32),
            None,
        )

    assert _contract_key(k_args(65536)) == _contract_key(k_args(131072))
    assert _bucket_key(k_args(65536)) == _bucket_key(k_args(131072))
    assert _bucket_key(k_args(32 * 1024)) != _bucket_key(k_args(32 * 1024 + 1))


def test_gemm_tune_and_contract_keys_exclude_tune_mode():
    args = _varlen_m_args(4096)
    base_w = _bucket_key(args)
    base_f = _contract_key(args)
    # Extra-kwargs collection path.
    assert _bucket_key(args, {"tune_mode": 1}) == base_w
    assert _contract_key(args, {"tune_mode": 1}) == base_f
    # Signature-scalar collection path: same synthetic signature with and
    # without the tune_mode parameter is the only allowed difference.
    from cute_gemm.varlen_selector import gemm_contract_key, gemm_tune_key

    plain_names = ("A", "B", "out", "C", "bias", "alpha", "beta", "cu_seqlens_m")
    mode_names = (*plain_names, "tune_mode")
    a, b, out, cu = args[0], args[1], args[2], args[7]
    base8 = (a, b, out, None, None, 1.0, 1.0, cu)
    assert gemm_tune_key((*base8, 1), {}, mode_names) == gemm_tune_key(base8, {}, plain_names)
    assert gemm_contract_key((*base8, 1), {}, mode_names) == gemm_contract_key(
        base8, {}, plain_names
    )


def test_gemm_contract_key_dense_returns_none():
    import torch

    dense = (torch.empty((128, 64)), torch.empty((64, 128)), torch.empty((128, 128)))
    assert _contract_key(dense) is None


def test_autotune_config_supports_multi_kwarg_hash_and_equality():
    config_a = AutotuneConfig(block_m=128, num_warps=4)
    config_b = AutotuneConfig(block_m=128, num_warps=4)
    config_c = AutotuneConfig(block_m=64, num_warps=4)

    assert config_a == config_b
    assert hash(config_a) == hash(config_b)
    assert config_a != config_c

    timings = {config_a: 1.25, config_c: 2.5}
    assert timings[config_b] == 1.25
    assert len({config_a, config_b, config_c}) == 2


def test_gemm_token_bucket_buckets_gather_indices_with_rows():
    import torch

    # MetaMoE EP8 routed-row sizes observed in training.  258012 is in the
    # 262144 bucket while 512303 is deliberately in the next bucket.
    common = dict(
        A=torch.empty((4096, 2048)),
        B=torch.empty((2048, 768)),
        cu_seqlens_m=torch.zeros((769,), dtype=torch.int32),
    )
    key_a = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((258012, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((258012,), dtype=torch.int32),
        ),
        {},
    )
    key_b = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((258013, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((258013,), dtype=torch.int32),
        ),
        {},
    )
    assert key_a == key_b

    key_a2 = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((300000, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((300000,), dtype=torch.int32),
        ),
        {},
    )
    assert key_a2 == key_a

    key_c = _bucket_key(
        (
            common["A"],
            common["B"],
            torch.empty((512303, 768)),
            None,
            None,
            1.0,
            1.0,
            common["cu_seqlens_m"],
            None,
            torch.empty((512303,), dtype=torch.int32),
        ),
        {},
    )
    assert key_c != key_a


def test_gemm_token_bucket_ep8_gather_rows_do_not_collide():
    import torch

    # With eight local experts, the routed row count is much larger than the
    # source table A. The key must use A_idx length for average-M bucketing.
    cu = torch.zeros((9,), dtype=torch.int32)
    source_a = torch.empty((4096, 2048))
    b = torch.empty((8, 2048, 768))

    def key_for(total_m):
        return _bucket_key(
            (
                source_a,
                b,
                torch.empty((total_m, 768)),
                None,
                None,
                1.0,
                1.0,
                cu,
                None,
                torch.empty((total_m,), dtype=torch.int32),
            ),
            {},
        )

    # Both averages exceed 1024: the open tail intentionally shares a key.
    assert key_for(258012) == key_for(512303)
    assert key_for(8 * 1024) != key_for(8 * 1024 + 1)


def test_gemm_token_bucket_preserves_static_n_for_varlen_k():
    import torch

    # For varlen-K, only A's K dimension and B's K dimension are dynamic.
    # out's N dimension is static and must remain exact in the key.
    a = torch.empty((2048, 258012))
    b = torch.empty((258012, 768))
    out_768 = torch.empty((2048, 768))
    out_1024 = torch.empty((2048, 1024))
    cu = torch.tensor([0, 258012], dtype=torch.int32)
    args = (a, b, out_768, None, None, 1.0, 1.0, None, cu, None)
    key_768 = _bucket_key(args, {})
    key_1024 = _bucket_key((a, b, out_1024, None, None, 1.0, 1.0, None, cu, None), {})
    assert key_768 != key_1024


def test_gemm_token_bucket_keeps_quantized_output_scale_shape():
    import torch

    # Quantized-output scale factors are named tensors, not operands. Two
    # different output-scale layouts must not share one tuning result.
    cu = torch.zeros((9,), dtype=torch.int32)
    base = (
        torch.empty((258012, 2048)),
        torch.empty((768, 2048, 1536)),
        torch.empty((258012, 1536)),
        None,
        None,
        1.0,
        1.0,
        cu,
        None,
        None,
    )
    key_a = _bucket_key(base, {"SFD": torch.empty((128, 4, 32, 4, 4))})
    key_b = _bucket_key(base, {"SFD": torch.empty((128, 8, 32, 4, 4))})
    assert key_a is not None
    assert key_a != key_b


def test_gemm_token_bucket_keeps_quantized_output_scale_shape_for_positional_tail():
    import torch

    # cu_seqlens_m passed positionally: the hook has to name the positional
    # arguments itself to recognise a varlen call at all, and the positional
    # tail (SFD at index 23) still has to reach the key.
    cu = torch.zeros((9,), dtype=torch.int32)
    prefix = (
        torch.empty((258012, 2048)),
        torch.empty((768, 2048, 1536)),
        torch.empty((258012, 1536)),
        None,
        None,
        1.0,
        1.0,
        cu,
        None,
        None,
        None,
        False,
        False,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        0,
    )
    assert len(prefix) == 23
    key_a = _bucket_key(prefix + (torch.empty((128, 4, 32, 4, 4)), None, None), {})
    key_b = _bucket_key(prefix + (torch.empty((128, 8, 32, 4, 4)), None, None), {})
    assert key_a is not None
    assert key_a != key_b


def test_gemm_token_bucket_keeps_average_expert_rows_bucket():
    import torch

    # Both totals are in the 262144 total-row bucket, but their average rows
    # per expert cross an average-M bucket and should not share tuning.
    cu = torch.empty((769,), dtype=torch.int32)
    common = (None, None, 1.0, 1.0, cu, None, None)
    key_small = _bucket_key(
        (
            torch.empty((100000, 2048)),
            torch.empty((768, 2048, 1536)),
            torch.empty((100000, 1536)),
            *common,
        ),
        {},
    )
    key_large = _bucket_key(
        (
            torch.empty((200000, 2048)),
            torch.empty((768, 2048, 1536)),
            torch.empty((200000, 1536)),
            *common,
        ),
        {},
    )
    assert key_small != key_large


def test_gemm_token_bucket_separates_same_bucket_across_selector_leaves():
    import torch

    # Both totals round to the same average-rows bucket (64 experts), but
    # K/N=78.125 and K/N=79.1015625 land on opposite K-group leaves.
    def key_for(total_k):
        a = torch.empty((2048, total_k), dtype=torch.bfloat16)
        b = torch.empty((total_k, 2048), dtype=torch.bfloat16)
        out = torch.empty((2048, 2048), dtype=torch.bfloat16)
        return _bucket_key(
            (a, b, out, None, None, 1.0, 1.0, None, torch.zeros((65,), dtype=torch.int32), None),
            {},
        )

    assert key_for(160000) != key_for(162000)


def test_gemm_token_key_uses_capability_not_device_ordinal(monkeypatch):
    """Equivalent CUDA ordinals share a winner; architecture still isolates it."""
    import torch

    import cute_gemm.varlen_selector as key_module

    a = torch.empty((4096, 2048), dtype=torch.bfloat16)
    b = torch.empty((2048, 1536), dtype=torch.bfloat16)
    out = torch.empty((4096, 1536), dtype=torch.bfloat16)
    cu = torch.empty((65,), dtype=torch.int32)
    monkeypatch.setattr(key_module, "_device_capability", lambda tensor: (9, 0))
    sm90 = _bucket_key((a, b, out, None, None, 1.0, 1.0, None, cu, None), {})
    assert ("device_capability", (9, 0)) in sm90
    assert not any(entry[0] == "device" for entry in sm90 if isinstance(entry, tuple))

    monkeypatch.setattr(key_module, "_device_capability", lambda tensor: (10, 0))
    sm100 = _bucket_key((a, b, out, None, None, 1.0, 1.0, None, cu, None), {})
    assert sm100 != sm90


def test_gemm_token_key_tracks_selector_switch_and_bench_budget(monkeypatch):
    import torch

    def key_for():
        a = torch.empty((2048, 160000), dtype=torch.bfloat16)
        b = torch.empty((160000, 2048), dtype=torch.bfloat16)
        out = torch.empty((2048, 2048), dtype=torch.bfloat16)
        cu = torch.zeros((65,), dtype=torch.int32)
        return _bucket_key((a, b, out, None, None, 1.0, 1.0, None, cu, None), {})

    enabled = key_for()
    monkeypatch.setenv("DLKERNEL_VARLEN_SELECTOR", "0")
    disabled = key_for()
    assert enabled != disabled

    monkeypatch.setenv("DLKERNEL_VARLEN_SELECTOR", "1")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "64")
    budget = key_for()
    assert budget != enabled


def test_gemm_token_key_normalizes_omitted_and_explicit_defaults():
    import torch

    from cute_gemm.gemm_interface import gemm_tuned
    from cute_gemm.varlen_selector import gemm_tune_key

    a = torch.empty((4096, 2048))
    b = torch.empty((2048, 1536))
    out = torch.empty((4096, 1536))
    cu = torch.zeros((65,), dtype=torch.int32)
    names = gemm_tuned.arg_names
    defaults = gemm_tuned.arg_defaults
    positional = gemm_tune_key((a, b, out), {"cu_seqlens_m": cu}, names, defaults)
    explicit = {
        name: default
        for name, default in zip(names, defaults)
        if default is not inspect.Signature.empty
    }
    explicit.update({"A": a, "B": b, "out": out, "cu_seqlens_m": cu})
    keyword = gemm_tune_key((), explicit, names, defaults)
    assert positional == keyword


def test_large_sm90_varlen_m_prunes_to_the_candidate_budget(monkeypatch):
    """End to end through the pruning hook: an in-envelope varlen call is
    reduced to the shipped budget, as a subset of the structural pool."""
    import torch

    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import _get_sm90_configs
    from cute_gemm.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    structural = set(range(len(configs)))
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((258012, 2048), dtype=torch.bfloat16),
            "B": torch.empty((768, 2048, 1536), dtype=torch.bfloat16),
            "out": torch.empty((258012, 1536), dtype=torch.bfloat16),
            "cu_seqlens_m": torch.zeros((769,), dtype=torch.int32),
        },
    )
    assert len(kept) == KEEP
    kept_ids = {id(conf) for conf in kept}
    assert kept_ids <= {id(conf) for conf in configs}
    assert len(structural) == len(configs)


def test_large_sm90_varlen_k_prunes_to_the_candidate_budget(monkeypatch):
    import torch

    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import _get_sm90_configs
    from cute_gemm.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((2048, 258012), dtype=torch.bfloat16),
            "B": torch.empty((258012, 768), dtype=torch.bfloat16),
            "out": torch.empty((2048, 768), dtype=torch.bfloat16),
            # Only prefix-sum length is inspected: 32 experts are in coverage.
            # Routed K comes from B.shape[0], not the stored values.
            "cu_seqlens_k": torch.arange(0, 33, dtype=torch.int32) * 1000,
        },
    )
    assert len(kept) == KEEP


def test_large_sm90_varlen_out_of_coverage_keeps_the_structural_pool(monkeypatch):
    """float32 is outside the measured coverage, so the selector must decline
    and hand the structural pool (22 M-group / 44 K-group + 2 dynamic twins)
    to the autotuner."""
    import torch

    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import _get_sm90_configs

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    kept_m = _select_via_policy(
        configs,
        {
            "A": torch.empty((196608, 2048)),
            "B": torch.empty((768, 2048, 1536)),
            "cu_seqlens_m": torch.zeros((769,), dtype=torch.int32),
        },
    )
    kept_k = _select_via_policy(
        configs,
        {
            "A": torch.empty((2048, 393216)),
            "B": torch.empty((393216, 768)),
            "cu_seqlens_k": torch.zeros((769,), dtype=torch.int32),
        },
    )
    assert len(kept_m) == 22  # varlen_m never gains the dynamic twins
    assert len(kept_k) == 46  # 44 static + 2 dynamic twins


def test_large_sm90_varlen_k_high_keeps_the_dynamic_twins(monkeypatch):
    """The k-high prune keeps the 6 static matches plus the measured
    dynamic-persistent twins; every other leaf rejects dynamic configs."""
    import torch

    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import _get_sm90_configs, get_sm90_dynamic_varlen_configs
    from cute_gemm.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    # the hook itself appends the twins to the static pool (production path)
    twin_configs = get_sm90_dynamic_varlen_configs()
    assert len(twin_configs) == 2
    kept = _select_via_policy(
        [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()],
        {
            "A": torch.empty((2048, 258012), dtype=torch.bfloat16),
            "B": torch.empty((258012, 768), dtype=torch.bfloat16),
            "out": torch.empty((2048, 768), dtype=torch.bfloat16),
            "cu_seqlens_k": torch.arange(0, 33, dtype=torch.int32) * 1000,
        },
    )
    assert len(kept) == KEEP
    kept_dynamic = [conf for conf in kept if conf.kwargs["config"].is_dynamic_persistent]
    assert {
        (conf.kwargs["config"].tile_m, conf.kwargs["config"].swap_ab) for conf in kept_dynamic
    } == {(256, False), (256, True)}


def test_large_sm90_varlen_selector_kill_switch(monkeypatch):
    """DLKERNEL_VARLEN_SELECTOR=0 restores the full structural pool."""
    import torch

    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import _get_sm90_configs
    from cute_gemm.varlen_selector import KEEP

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    monkeypatch.setenv("DLKERNEL_VARLEN_SELECTOR", "0")
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((2048, 258012), dtype=torch.bfloat16),
            "B": torch.empty((258012, 768), dtype=torch.bfloat16),
            "out": torch.empty((2048, 768), dtype=torch.bfloat16),
            "cu_seqlens_k": torch.arange(0, 33, dtype=torch.int32) * 1000,
        },
    )
    assert len(kept) > KEEP


def test_large_sm90_gather_prune_keeps_structural_filters(monkeypatch):
    """gather_A is outside the selector's coverage, so the structural rules are
    the only thing pruning: cluster_n == 1, no swap_ab, and use_tma_gather is
    rejected below SM100."""
    from dataclasses import replace

    import torch

    import cute_gemm.gemm_interface as gi
    from cute_gemm.gemm_config import _get_sm90_configs

    monkeypatch.setattr(gi, "get_device_capacity", lambda device: (9, 0))
    configs = [gi.AutotuneConfig(config=c) for c in _get_sm90_configs()]
    structural = next(
        c for c in configs if c.kwargs["config"].cluster_n == 1 and not c.kwargs["config"].swap_ab
    )
    configs.append(
        gi.AutotuneConfig(config=replace(structural.kwargs["config"], use_tma_gather=True))
    )
    kept = _select_via_policy(
        configs,
        {
            "A": torch.empty((4096, 2048)),
            "cu_seqlens_m": torch.zeros((9,), dtype=torch.int32),
            "A_idx": torch.empty((258012,), dtype=torch.int32),
        },
    )
    assert kept
    assert all(c.kwargs["config"].cluster_n == 1 for c in kept)
    assert all(not c.kwargs["config"].swap_ab for c in kept)
    assert all(not c.kwargs["config"].use_tma_gather for c in kept)


def test_autotune_bench_loop_defers_and_retries(monkeypatch):
    """A config whose kernel raises CompilePending is rotated to the back and
    retried once its sha polls done; all configs end up benchmarked exactly
    as if they had been warm.

    This encodes the compile-only rip-out contract: the autotuner no longer
    precompiles via fake tensors — the bench loop discovers cold keys with
    the real tensors in-process and overlaps compilation via the pool.
    """
    from cute_gemm.autotuner import AutotuneConfig, Autotuner
    from cute_gemm.cache import async_compile
    from cute_gemm.cache.async_compile import CompilePending

    class _StubPool:
        """poll() reports 'pending' once per sha, then 'done'."""

        def __init__(self):
            self.polls = {}

        def poll(self, sha):
            n = self.polls.get(sha, 0) + 1
            self.polls[sha] = n
            return ("pending" if n == 1 else "done"), None

    stub = _StubPool()
    monkeypatch.setattr(async_compile, "_active_pool", stub)

    bench_order = []
    raised_once = set()

    def kernel(x, block: int = 0):
        # config block=1 is "cold": its first invocation defers.
        if block == 1 and 1 not in raised_once:
            raised_once.add(1)
            raise CompilePending("f" * 64, "fake._compile_kernel")
        bench_order.append(block)

    def do_bench(fn, quantiles=None, **kw):
        fn()
        return [1.0 + bench_order[-1], 1.0, 1.0]  # block=0 fastest

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1, 2)],
        do_bench=do_bench,
    )
    import torch

    x = torch.empty(4, device="cuda")
    tuner(x)

    # Bench order: block=1 deferred, so it benched AFTER block 2 (exactly
    # once). The trailing 0 is __call__'s real invocation with the winner.
    assert bench_order == [0, 2, 1, 0], bench_order
    assert stub.polls == {"f" * 64: 2}  # one rotation, one release
    assert len(tuner.configs_timings) == 3
    best = tuner.cache[next(iter(tuner.cache))]
    assert best.kwargs["block"] == 0  # timings intact despite the deferral


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="_gpu_warmup needs a GPU")
def test_autotune_wedged_pool_falls_back_in_process(monkeypatch):
    """A sha that never resolves must not hang the sweep: past the attempt
    cap the config is benched with the pool suppressed (in-process compile),
    so autotuning always terminates.
    """
    import cute_gemm.autotuner as at
    from cute_gemm.autotuner import AutotuneConfig, Autotuner
    from cute_gemm.cache import async_compile
    from cute_gemm.cache.async_compile import CompilePending, get_active_pool

    class _WedgedPool:
        def poll(self, sha):
            return "pending", None  # never completes

    monkeypatch.setattr(async_compile, "_active_pool", _WedgedPool())
    monkeypatch.setattr(at, "_POOL_WEDGE_TIMEOUT_S", 0.2)

    benched = []

    def kernel(x, block: int = 0):
        # Defer as long as a pool is visible; succeed once suppressed.
        if block == 1 and get_active_pool() is not None:
            raise CompilePending("e" * 64, "fake._compile_kernel")
        benched.append(block)

    tuner = Autotuner(
        kernel,
        key=[],
        configs=[AutotuneConfig(block=b) for b in (0, 1)],
        do_bench=lambda fn, quantiles=None, **kw: (fn(), [1.0, 1.0, 1.0])[1],
    )
    import torch

    tuner(torch.empty(4, device="cuda"))
    assert benched.count(1) == 1  # eventually ran, via suppress_pool
    assert len(tuner.configs_timings) == 2


def test_gemm_tune_key_declines_for_dense_calls():
    import torch

    from cute_gemm.gemm_interface import gemm_tuned
    from cute_gemm.varlen_selector import gemm_tune_key

    # No cu_seqlens anywhere: the hook must opt out so the autotuner keeps its
    # own default key for dense GEMM.
    args = (
        torch.empty((128, 64)),
        torch.empty((64, 128)),
        torch.empty((128, 128)),
        None,
        None,
        1.0,
        1.0,
        None,
        None,
        None,
    )
    assert gemm_tune_key(args, {}, gemm_tuned.arg_names) is None
    assert (
        gemm_tune_key(args[:7], {"cu_seqlens_m": None, "cu_seqlens_k": None}, gemm_tuned.arg_names)
        is None
    )


def test_gemm_tune_key_buckets_varlen_arguments_passed_by_keyword():
    import torch

    from cute_gemm.gemm_interface import gemm_tuned
    from cute_gemm.varlen_selector import gemm_tune_key

    # cute_gemm/gemm.py and the epilogue frontend pass cu_seqlens by keyword, so
    # the hook must recognise that form and bucket it exactly like the
    # positional one.
    def key_for(total_k, total_m=None):
        a = torch.empty((2048, total_k))
        b = torch.empty((total_k, 768))
        out = torch.empty((2048 if total_m is None else total_m, 768))
        kwargs = {"cu_seqlens_k": torch.zeros((9,), dtype=torch.int32)}
        return gemm_tune_key((a, b, out, None, None, 1.0, 1.0), kwargs, gemm_tuned.arg_names)

    assert key_for(258012) is not None
    # Both routed averages land in the open >1024 tail.
    assert key_for(258012) == key_for(258013)
    assert key_for(258012) == key_for(200000)
    assert key_for(512303) == key_for(258012)
    assert key_for(8 * 1024) != key_for(8 * 1024 + 1)


def test_bench_budget_override_is_gated_on_the_varlen_key_hook(monkeypatch):
    """Only a call that opted into the coarse key may change the bench budget.

    The autotuner object is shared by dense and routed GEMM, so the knob must
    not leak: a dense tune keeps the historical 200 ms / 200-call protocol even
    when the environment asks for a shorter cold-start sweep.
    """
    from cute_gemm.autotuner import _DEFAULT_BENCH_TIMED_CALLS, _DEFAULT_BENCH_WARMUP_MS, Autotuner

    tuner = Autotuner(lambda *a, **k: None, [], [AutotuneConfig(tile_m=128)])
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "40")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "32")

    # dense: hook declines -> historical protocol, environment ignored
    tuner._budget_override = False
    assert tuner._bench_budget() == (_DEFAULT_BENCH_WARMUP_MS, _DEFAULT_BENCH_TIMED_CALLS)

    # routed: hook returns a key -> the override applies, clamped into range
    tuner._budget_override = True
    assert tuner._bench_budget() == (40.0, 32)

    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "4")
    assert tuner._bench_budget() == (40.0, 16)  # floor keeps the sweep meaningful


def test_call_sets_the_budget_flag_from_the_policy(monkeypatch):
    """``__call__`` derives the flag from the policy, not from stale state.

    Both directions are pinned with a warm cache, so no kernel is ever benched.
    """
    import torch

    from cute_gemm.autotuner import AutotuneConfig, Autotuner, TunePolicy

    configs = [AutotuneConfig(x=1), AutotuneConfig(x=2)]
    calls = []

    class DecliningPolicy(TunePolicy):
        # Default key, no budget opt-in: the plain-operator specialization.
        def make_key(self, args, kwargs, default_key):
            calls.append(1)
            return default_key()

    tuner = Autotuner(lambda *a, **k: None, [], configs, policy=DecliningPolicy())
    tensor = torch.empty(4, 4)
    default_key = ((4, 4), (2, 1), tensor.dtype)  # what __call__ builds for a bare tensor
    tuner.cache[default_key] = configs[0]
    tuner._budget_override = True  # stale value from an earlier routed call
    tuner(tensor)
    assert calls == [1]
    assert tuner._budget_override is False

    class RoutedPolicy(TunePolicy):
        # Produces its own bucket-style key and opts into the routed budget.
        def make_key(self, args, kwargs, default_key):
            return ("routed",)

        def budget_override(self, key):
            return True

    routed = Autotuner(lambda *a, **k: None, [], configs, policy=RoutedPolicy())
    routed.cache[("routed",)] = configs[0]
    routed(tensor)
    assert routed._budget_override is True


def test_bench_budget_rejects_invalid_environment_values(monkeypatch):
    from cute_gemm.autotuner import AutotuneConfig, Autotuner

    tuner = Autotuner(lambda x, **kwargs: None, [], [AutotuneConfig(x=1)])
    tuner._budget_override = True
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "nan")
    with pytest.raises(ValueError, match="DLKERNEL_TUNE_WARMUP_MS"):
        tuner._bench_budget()
    monkeypatch.setenv("DLKERNEL_TUNE_WARMUP_MS", "40")
    monkeypatch.setenv("DLKERNEL_TUNE_TIMED_CALLS", "oops")
    with pytest.raises(ValueError, match="DLKERNEL_TUNE_TIMED_CALLS"):
        tuner._bench_budget()


# ---------------------------------------------------------------------------
# TuneMode chain semantics (v4 plan §2.2): measured winners are shared across
# modes; STATIC never benches; RUNNING tunes once per fixed contract F.  The
# harness drives W and F off different tensor axes so the two identities vary
# independently: x = torch.empty(n) gives W = n // 10 and F = n % 10.
# ---------------------------------------------------------------------------


def _mode_harness():

    from cute_gemm.autotuner import AutotuneConfig, Autotuner, TunePolicy

    class Recorder:
        fn_configs = None
        bench_calls = 0

    rec = Recorder()
    rec.fn_configs = []

    def fn(x, config=None, **kw):
        rec.fn_configs.append(config)

    def do_bench(kernel_call, quantiles=None, **kw):
        kernel_call()
        rec.bench_calls += 1
        return [1.0, 1.0, 1.0]

    class WF_Policy(TunePolicy):
        def make_key(self, args, kwargs, default_key):
            return ("W", args[0].shape[0] // 10)

        def make_contract(self, args, kwargs, default_key):
            return ("F", args[0].shape[0] % 10)

    tuner = Autotuner(
        fn,
        [],
        [AutotuneConfig(config=1), AutotuneConfig(config=2)],
        do_bench=do_bench,
        policy=WF_Policy(),
    )
    return tuner, rec


@pytest.fixture(autouse=True)
def _no_gpu_warmup(monkeypatch):
    """Mode tests exercise dispatch, not the bench protocol itself."""
    import cute_gemm.autotuner as at

    monkeypatch.setattr(at, "_gpu_warmup", lambda duration_ms=200: None)


def test_static_miss_never_benches_and_falls_back(monkeypatch):
    import torch

    from cute_gemm.gemm_config import TuneMode

    tuner, rec = _mode_harness()
    with pytest.warns(RuntimeWarning, match="analytical fallback"):
        tuner(torch.empty(17), tune_mode=TuneMode.STATIC)
    assert rec.bench_calls == 0
    assert rec.fn_configs == [None]  # no config injected: the -1 call
    assert not tuner.cache  # no winner manufactured, nothing persisted

    # Same key again: still correct, but the warning is one-shot.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tuner(torch.empty(17), tune_mode=TuneMode.STATIC)
    assert rec.fn_configs == [None, None]


def test_measured_winners_are_shared_across_modes():
    import torch

    from cute_gemm.gemm_config import TuneMode

    tuner, rec = _mode_harness()
    tuner(torch.empty(17), tune_mode=TuneMode.TUNED)  # benches, winner cached
    assert rec.bench_calls == 2  # one sweep = 2 candidates
    assert tuner.cache[("W", 1)].kwargs["config"] == 1

    tuner(torch.empty(17), tune_mode=TuneMode.STATIC)  # memory hit: winner reused
    assert rec.bench_calls == 2
    assert rec.fn_configs[-1] == 1


def test_running_tunes_once_per_contract():
    import torch

    from cute_gemm.gemm_config import TuneMode

    tuner, rec = _mode_harness()
    tuner(torch.empty(17), tune_mode=TuneMode.RUNNING)  # W1, F7: first tune
    assert rec.bench_calls == 2  # one sweep = 2 candidates
    tuner(torch.empty(27), tune_mode=TuneMode.RUNNING)  # W2, F7: spent -> reuse F7 winner
    assert rec.bench_calls == 2
    # 2026-10-22 design: the contract's first online tune transfers across
    # token buckets -- the fn runs with the F7 winner, explicit.
    assert rec.fn_configs[-1] == 1
    # New bucket AND new contract (38 -> W3, F8): the memo is keyed by W, and
    # production policies derive F from the same scan as W (same W => same F),
    # so a fresh contract always arrives under a fresh W.  It tunes.
    tuner(torch.empty(38), tune_mode=TuneMode.RUNNING)
    assert rec.bench_calls == 4
    tuner(torch.empty(17), tune_mode=TuneMode.RUNNING)  # W1: memory winner
    assert rec.bench_calls == 4
    assert rec.fn_configs[-1] == 1


def test_tuned_still_benches_new_w_and_shares_the_record():
    import torch

    from cute_gemm.gemm_config import TuneMode

    tuner, rec = _mode_harness()
    tuner(torch.empty(17), tune_mode=TuneMode.RUNNING)
    tuner(torch.empty(27), tune_mode=TuneMode.TUNED)  # F7 measured, new W: TUNED re-benches
    assert rec.bench_calls == 4
    # The TUNED sweep shares its record: RUNNING on a third W of F7 reuses
    # the contract winner (2026-10-22) instead of the analytical fallback.
    tuner(torch.empty(37), tune_mode=TuneMode.RUNNING)
    assert rec.bench_calls == 4
    assert rec.fn_configs[-1] == 1


def test_missing_tune_mode_keeps_legacy_bench():
    import torch

    tuner, rec = _mode_harness()
    tuner(torch.empty(17))
    tuner(torch.empty(27))
    assert rec.bench_calls == 4  # every new W sweeps (2 candidates each): today's behaviour
    assert rec.fn_configs[-1] == 1


def test_varlen_selector_accepts_canonical_sm_arch_override(monkeypatch):
    import torch

    import cute_gemm.varlen_selector as selector_mod
    from cute_gemm.autotuner import AutotuneConfig
    from cute_gemm.varlen_selector import VarlenGemmTunePolicy

    seen = []

    def selector(configs, **kwargs):
        seen.append(kwargs["device_capability"])
        return configs

    monkeypatch.setattr(selector_mod, "select_varlen_gemm_candidates", selector)
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(("A", "B", "cu_seqlens_m"), (None,) * 3)
    monkeypatch.setenv("DLKERNEL_ARCH", "sm_90")
    A = torch.empty((64, 16), dtype=torch.bfloat16)
    B = torch.empty((2, 16, 32), dtype=torch.bfloat16)
    cu = torch.empty((3,), dtype=torch.int32)
    configs = [AutotuneConfig(config="a")]
    assert policy.shortlist(configs, {"A": A, "B": B}, {"cu_seqlens_m": cu}) == configs
    assert seen == [9]


# ---------------------------------------------------------------------------
# Autotuner policy/cache contracts (CPU-only)
# ---------------------------------------------------------------------------


def test_winner_cache_ignores_object_cache_dir(tmp_path, monkeypatch):
    """``FileCacheManager`` (winner JSONs) keys off ``DLKERNEL_CACHE_DIR`` only.

    ``DLKERNEL_ARTIFACT_CACHE_DIR`` exists so multi-process workloads can share
    one ``.o`` store while keeping tuning measurements hermetic; if it ever
    leaked into the winner cache, per-job benchmark isolation would silently
    break. Pure path logic — no CUDA.
    """
    from cute_gemm.autotuner import FileCacheManager

    tune_root = tmp_path / "tune"
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tune_root))
    monkeypatch.setenv("DLKERNEL_ARTIFACT_CACHE_DIR", str(tmp_path / "shared-objects"))

    manager = FileCacheManager("deadbeef00")

    assert manager.cache_dir.startswith(str(tune_root)), manager.cache_dir
    assert "shared-objects" not in manager.cache_dir


@pytest.fixture
def _cpu_tuning(monkeypatch, tmp_path):
    """Disable GPU-only setup while exercising the host-side tuner contract."""
    from contextlib import nullcontext

    import cute_gemm.autotuner as at
    from cute_gemm.cache import async_compile

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DLKERNEL_FORCE_CACHE_UPDATE", raising=False)
    monkeypatch.setattr(at, "_gpu_warmup", lambda: None)
    monkeypatch.setattr(async_compile, "pool_scope", lambda: nullcontext(None))


def _bench_once(fn, **kwargs):
    fn()
    return [1.0, 1.0, 1.0]


@pytest.mark.parametrize("direction", ["cu_seqlens_m", "cu_seqlens_k"])
@pytest.mark.parametrize("positional", [False, True])
def test_autotuner_required_cache_rejects_varlen_miss(
    _cpu_tuning, monkeypatch, direction, positional
):
    import torch

    import cute_gemm.autotuner as at

    monkeypatch.setenv("DLKERNEL_REQUIRE_TUNE_CACHE", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("required-cache miss entered warmup or benchmarking")

    monkeypatch.setattr(at, "_gpu_warmup", forbidden)

    def kernel(x, cu_seqlens_m=None, cu_seqlens_k=None, block=1):
        return x * block

    tuner = at.Autotuner(
        kernel,
        configs=[at.AutotuneConfig(block=b) for b in (1, 2)],
        key=[],
        do_bench=forbidden,
        cache_results=True,
    )
    x = torch.arange(4.0)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Missing autotune cache"):
        if positional:
            tuner(
                x,
                cu if direction == "cu_seqlens_m" else None,
                cu if direction == "cu_seqlens_k" else None,
            )
        else:
            tuner(x, **{direction: cu})


def test_autotuner_cache_miss_tunes_when_required_cache_is_unset(_cpu_tuning, monkeypatch):
    import torch

    import cute_gemm.autotuner as at

    monkeypatch.delenv("DLKERNEL_REQUIRE_TUNE_CACHE", raising=False)

    def kernel(x, block=1):
        return x * block

    tuner = at.Autotuner(
        kernel,
        configs=[at.AutotuneConfig(block=b) for b in (1, 2)],
        key=[],
        do_bench=_bench_once,
        cache_results=True,
    )
    x = torch.arange(4.0)
    torch.testing.assert_close(tuner(x), x)


def test_autotuner_does_not_cache_an_all_failed_sweep(_cpu_tuning):
    import torch

    import cute_gemm.autotuner as at

    def failed_bench(fn, **kwargs):
        fn()
        return [float("inf")] * 3

    def kernel(x, block=1):
        return x * block

    tuner = at.Autotuner(
        kernel,
        configs=[at.AutotuneConfig(block=b) for b in (1, 2)],
        key=[],
        do_bench=failed_bench,
        cache_results=True,
    )
    with pytest.raises(RuntimeError, match="all .* candidates failed"):
        tuner(torch.arange(4.0))
    assert tuner.cache == {}


class _RecordingPolicy(TunePolicy):
    """Small policy test double that records the public hook order."""

    def __init__(self, events):
        self.events = events

    def make_key(self, args, kwargs, default_key):
        self.events.append("key")
        return ("policy", default_key())

    def prepare_candidates(self, configs, named_args, kwargs):
        self.events.append("prepare")
        return list(configs)

    def shortlist(self, configs, named_args, kwargs):
        self.events.append("shortlist")
        return list(configs[:1])


def test_autotuner_policy_hooks_run_in_order():
    import torch

    from cute_gemm.autotuner import AutotuneConfig, Autotuner

    events = []

    def kernel(x, scale=1, block=1):
        return x * scale * block

    tuner = Autotuner(
        kernel,
        configs=[AutotuneConfig(block=1), AutotuneConfig(block=2)],
        key=["scale"],
        policy=_RecordingPolicy(events),
        do_bench=_bench_once,
    )
    x = torch.arange(4.0)
    torch.testing.assert_close(tuner(x, scale=3), x * 3)
    assert events == ["key", "prepare", "shortlist"]


def test_autotuner_policy_rejects_empty_or_duplicate_candidates():
    import torch

    from cute_gemm.autotuner import AutotuneConfig, Autotuner, TunePolicy

    configs = [AutotuneConfig(x=1), AutotuneConfig(x=2)]

    class EmptyPolicy(TunePolicy):
        def prepare_candidates(self, configs, named_args, kwargs):
            return []

    empty = Autotuner(
        lambda x, config=None: None,
        [],
        configs,
        policy=EmptyPolicy(),
        do_bench=lambda *args, **kwargs: pytest.fail("benchmark must not run"),
    )
    with pytest.raises(RuntimeError, match="empty candidate"):
        empty(torch.empty(1))

    class DuplicatePolicy(TunePolicy):
        def shortlist(self, configs, named_args, kwargs):
            return [configs[0], configs[0]]

    duplicate = Autotuner(
        lambda x, config=None: None,
        [],
        configs,
        policy=DuplicatePolicy(),
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        duplicate(torch.empty(1))


def test_gemm_policy_keeps_split_k_legality_separate_from_shortlist(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from cute_gemm.autotuner import AutotuneConfig
    from cute_gemm.gemm_config import GemmConfig
    from cute_gemm.gemm_tune_policy import GemmTunePolicy

    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )
    policy = GemmTunePolicy()
    config = GemmConfig(device_capacity=9)
    base = [AutotuneConfig(config=config)]
    named = {"A": torch.empty((4096, 4096)), "B": torch.empty((4096, 4096))}

    prepared = policy.prepare_candidates(base, named, {"split_k": None})
    assert prepared == [
        *base,
        *(AutotuneConfig(config=replace(config, split_k=s)) for s in (2, 4, 8, 16)),
    ]
    # Occupancy is a shortlist concern; this shape is not occupancy-starved.
    assert policy.shortlist(prepared, named, {"split_k": None}) == base


def test_tune_cache_lock_times_out_and_recovers(tmp_path, monkeypatch):
    """A frozen lock holder (alive but not moving) must not wedge the tuner:
    the disk-cache flock is bounded (LOCK_NB + timeout -> RuntimeError,
    flashinfer-aligned), and killing the holder lets the next call through.
    RED evidence: repro_flock_wedge.py froze a victim indefinitely at
    autotuner.py:463 before this change."""
    import subprocess
    import sys
    import time

    from cute_gemm.autotuner import AutotuneConfig, TunePolicy, autotune
    from cute_gemm.cache import jit

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DLKERNEL_TUNE_LOCK_TIMEOUT_S", "2")

    class P(TunePolicy):
        def make_key(self, args, kwargs, default_key):
            return ("lock-timeout",)

    bench_order = []  # one entry per do_bench call, i.e. per candidate swept

    def fixed_bench(fn, **kwargs):
        # Equal-cost candidates: a real bench picks the winner by noise (a
        # tie makes min() take the first), so pin config 2 to win instead.
        bench_order.append(None)
        return [float(3 - len(bench_order))]

    @autotune(
        configs=[AutotuneConfig(config=1), AutotuneConfig(config=2)],
        key=[],
        policy=P(),
        cache_results=True,
        do_bench=fixed_bench,
    )
    def victim_fn(x, config=None):
        return x * config

    real_lock = jit.FileLock
    paths = []

    class SpyLock(real_lock):
        def __init__(self, *a, **k):
            if a:
                paths.append(str(a[0]))
            super().__init__(*a, **k)

    monkeypatch.setattr(jit, "FileLock", SpyLock)
    assert victim_fn(3) == 6  # discover: bench + publish, records the lock path
    lock_path = paths[-1]
    for p in tmp_path.rglob("*.autotune.json"):  # 让下一次调用重新走冷路径
        p.unlink()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time; f = open(sys.argv[1], 'a+'); "
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX); print('HELD', flush=True); "
            "time.sleep(3600)",
            lock_path,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "HELD"
        victim_fn.cache.clear()  # 进程内 winner 也要清,才走磁盘路径
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="Timed out"):
            victim_fn(3)
        dt = time.monotonic() - t0
        assert 1.5 <= dt <= 15, f"lock timeout fired at {dt:.1f}s"
    finally:
        holder.kill()
        holder.wait()

    assert victim_fn(3) == 6  # 持锁者死亡 → flock 释放 → 正常完成


def test_running_disk_hit_does_not_consume_quota(tmp_path, monkeypatch):
    """C5 (plan v4 §6): a winner read from disk — including this process's
    own earlier sweep republished — is not a measurement; the contract keeps
    its first in-process sweep available for a NEW winner key."""
    import torch

    from cute_gemm.autotuner import AutotuneConfig, Autotuner, TunePolicy
    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    benched = [0]

    def fn(x, config=None, **kw):
        return x

    def do_bench(kernel_call, quantiles=None, **kw):
        kernel_call()
        benched[0] += 1
        return [1.0, 1.0, 1.0]

    class WF_Policy(TunePolicy):
        def make_key(self, args, kwargs, default_key):
            return ("W", args[0].shape[0] // 10)

        def make_contract(self, args, kwargs, default_key):
            return ("F", args[0].shape[0] % 10)

    def make_tuner():
        return Autotuner(
            fn,
            [],
            [AutotuneConfig(config=1), AutotuneConfig(config=2)],
            do_bench=do_bench,
            policy=WF_Policy(),
            cache_results=True,
        )

    writer = make_tuner()
    writer(torch.empty(17), tune_mode=TuneMode.RUNNING)  # sweep + publish
    assert benched[0] == 2

    reader = make_tuner()  # fresh memory, same disk cache dir
    reader(torch.empty(17), tune_mode=TuneMode.RUNNING)  # disk hit, no sweep
    assert benched[0] == 2
    assert ("W", 1) in reader.cache

    # Same contract F7, new winner key W2: the quota is still there.
    reader(torch.empty(27), tune_mode=TuneMode.RUNNING)
    assert benched[0] == 4


def test_pinned_config_bypasses_tuning_entirely(tmp_path, monkeypatch):
    """v4 §1 (review F2, 2026-10-21): a pinned config never sweeps, never
    consults the winner cache, and reaches the op verbatim — regardless of
    TuneMode and regardless of any previously measured winners."""
    import torch

    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    tuner, rec = _mode_harness()

    # Warm the winner cache first: a TUNED sweep stores W1.
    tuner(torch.empty(17), tune_mode=TuneMode.TUNED)
    assert rec.bench_calls > 0
    benches_before = rec.bench_calls

    # Pinned call on the SAME key: the pin must bypass everything — no bench,
    # and the op receives the pinned config object itself.
    rec.fn_configs.clear()
    tuner(torch.empty(17), tune_mode=TuneMode.TUNED, config=99)
    assert rec.bench_calls == benches_before
    assert rec.fn_configs == [99]

    # And on a cold key: still no sweep, still the pin.
    tuner(torch.empty(29), tune_mode=TuneMode.RUNNING, config=98)
    assert rec.bench_calls == benches_before
    assert rec.fn_configs == [99, 98]


def test_epi_minus_one_delegation_matches_low_level_signature():
    """The epi ``-1`` delegation targets the LOW-LEVEL ``mod.gemm`` (tile
    parameters only — no tune_mode/config, no tuner entry).  The stub below
    mirrors that exact signature with NO **kw, so any stray kwarg the
    delegation tries to pass (a 2026-10-21 fix once forwarded tune_mode and
    would have TypeError'd every real epi fallback) fails this test."""
    from cute_gemm.gemm_runtime.autotune import _make_tuned_fn

    calls = []

    class StubMod:
        sinks = {}
        _ident = "stub_mod"

        def gemm(
            self,
            A,
            B,
            D,
            C=None,
            *,
            epi_args,
            b_kn,
            cu_seqlens_m,
            A_idx,
            SFA,
            SFB,
            bs_format_a,
            bs_format_b,
            concat_layout,
            transform_a,
            transform_sf,
        ):
            calls.append(1)
            return "resolved"

    fn = _make_tuned_fn(StubMod(), epi_names=[])
    assert fn(None, None, None, None, config=None, tune_mode=2) == "resolved"
    assert calls == [1]


def test_failed_sweep_does_not_consume_running_quota(tmp_path, monkeypatch):
    """C5: only a sweep THIS process completed consumes the quota -- a failed
    sweep (all candidates error) raises and leaves the contract unmeasured."""
    import torch

    from cute_gemm.autotuner import AutotuneConfig, Autotuner
    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))

    def fn(x, config=None, **kw):
        return "ran"

    def bad_bench(kernel_call, quantiles=None, **kw):
        raise RuntimeError("candidate exploded")

    class P(TunePolicy):
        def make_key(self, args, kwargs, default_key):
            return ("W", args[0].shape[0] // 10)

        def make_contract(self, args, kwargs, default_key):
            return ("F", args[0].shape[0] % 10)

    tuner = Autotuner(
        fn,
        [],
        [AutotuneConfig(config=1), AutotuneConfig(config=2)],
        do_bench=bad_bench,
        policy=P(),
        cache_results=False,
    )
    with pytest.raises(RuntimeError):
        tuner(torch.empty(27), tune_mode=TuneMode.RUNNING)
    assert tuner._measured_contracts == set()


def test_force_cache_update_forces_resweep_on_hit(tmp_path, monkeypatch):
    """C7: DLKERNEL_FORCE_CACHE_UPDATE makes a disk winner a miss again -- the
    reader re-sweeps instead of adopting the entry (but the mode still
    governs: STATIC never sweeps, it falls back)."""
    import torch

    from cute_gemm.autotuner import AutotuneConfig, Autotuner
    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))

    def make_tuner():
        bench = {"n": 0}

        def fn(x, config=None, **kw):
            return "ran"

        def do_bench(kernel_call, quantiles=None, **kw):
            kernel_call()
            bench["n"] += 1
            return [1.0, 1.0, 1.0]

        class P(TunePolicy):
            def make_key(self, args, kwargs, default_key):
                return ("W", args[0].shape[0] // 10)

            def make_contract(self, args, kwargs, default_key):
                return ("F", args[0].shape[0] % 10)

        return Autotuner(
            fn,
            [],
            [AutotuneConfig(config=1), AutotuneConfig(config=2)],
            do_bench=do_bench,
            policy=P(),
            cache_results=True,
        ), bench

    writer, wb = make_tuner()
    writer(torch.empty(17), tune_mode=TuneMode.TUNED)
    assert wb["n"] > 0

    reader, rb = make_tuner()  # fresh memory, same disk dir
    reader(torch.empty(17), tune_mode=TuneMode.TUNED)
    assert rb["n"] == 0  # disk hit adopted

    monkeypatch.setenv("DLKERNEL_FORCE_CACHE_UPDATE", "1")
    forced, fb = make_tuner()
    forced(torch.empty(17), tune_mode=TuneMode.TUNED)
    assert fb["n"] > 0  # entry not adopted; swept again


def test_fallback_provenance_flag_drives_plan_cache_decision(tmp_path, monkeypatch):
    """P7 (2026-10-21): ``used_fallback`` is True only when the -1 fallback
    resolved the call -- winner hits and sweeps leave it False, so plan caches
    can distinguish 'safe to pin' from 'a peer may still publish a winner'."""
    import torch

    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    tuner, rec = _mode_harness()

    tuner(torch.empty(17), tune_mode=TuneMode.TUNED)  # sweep -> winner
    assert tuner.used_fallback is False
    tuner(torch.empty(17), tune_mode=TuneMode.TUNED)  # memory hit
    assert tuner.used_fallback is False
    tuner(torch.empty(29), tune_mode=TuneMode.STATIC)  # cold key -> -1
    assert tuner.used_fallback is True
    tuner(torch.empty(17), tune_mode=TuneMode.TUNED)  # winner again
    assert tuner.used_fallback is False


# ---------------------------------------------------------------------------
# -1 fallback sink hand-off (2026-10-21 fix): the wrapper's fallback resolves
# a config through the analytical ranker and slices the worst-case sink
# buffers to it, but ``tuned_mod_gemm`` used to return an empty ``sinks`` dict
# -- the eager frontend indexes ``TunedModGemm.sinks`` by its owned sinks and
# KeyError'd.  The wrapper now stashes what it ran with on the per-mod fn.
# CPU-only: the SM90 ranker is GPU-blind under DLKERNEL_ARCH and STATIC never
# benches.
# ---------------------------------------------------------------------------


def _stub_sink_mod(ident):
    """A signature-faithful mod double: one config-shaped (dim==0) sink, the
    low-level ``gemm`` entry (tile parameters only, no tuner kwargs), and the
    default-config resolution the -1 chain falls back to."""

    from cute_gemm.epilogue.frontend import EpiMod

    class _SinkOp:
        dim = 0

        @staticmethod
        def sink_alloc_shape(lead, n, tile_m, tile_n, num_seqs=None):
            del tile_m, num_seqs
            return (lead[-1], -(-n // tile_n))

    class _StubSinkMod:
        mode = "element"
        outputs = ()
        # Follow the current host preflight protocol. This double declares
        # one sink and no column-vector inputs.
        operand_names = ()
        _colvec_rows = staticmethod(EpiMod._colvec_rows)
        _preflight_colvecs = EpiMod._preflight_colvecs
        _ident = ident

        def __init__(self):
            self.gemm_calls = []
            self.sinks = {"stat": _SinkOp()}

        @property
        def semantic_digest(self):
            return f"{ident}-digest"

        @staticmethod
        def _default_config(A, B, transform_a):
            del A, B, transform_a
            from cute_gemm.gemm_config import GemmConfig

            return GemmConfig(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, device_capacity=9)

        def gemm(self, A, B, D, C=None, *, epi_args, **kwargs):
            self.gemm_calls.append((kwargs, dict(epi_args)))
            return "plan-sentinel"

    return _StubSinkMod()


def test_fallback_returns_ranked_sliced_sinks(_cpu_tuning, monkeypatch):
    """The ranked -1 call hands its SLICED sink views back: TunedModGemm.sinks
    carries the per-config views of the worst-case buffers, and ``config``
    stays None (the caller's winner-vs-fallback provenance signal)."""
    import torch

    import cute_gemm.gemm_runtime.autotune as epi_autotune
    from cute_gemm import gemm_heuristic
    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setattr(epi_autotune, "_MOD_TUNERS", {})
    m, k, n, experts = 4096, 512, 2048, 8
    A = torch.empty((m, k), dtype=torch.bfloat16)
    B = torch.empty((k, n), dtype=torch.bfloat16)  # b_kn: (K, N)
    D = torch.empty((m, n), dtype=torch.bfloat16)
    cu = torch.zeros((experts + 1,), dtype=torch.int32)
    mod = _stub_sink_mod("ranked_sink_mod")
    worst = epi_autotune.sink_arg_shapes(mod, m, n, device="cpu")
    buf = torch.empty(worst["stat"], dtype=torch.float32)

    # Self-check: this varlen_m desc is ranker-covered, and the ranked config
    # decides the expected sink shape below.
    desc = gemm_heuristic.desc_from_call(A, B, D, cu, None)
    ranked = gemm_heuristic.rank(desc, gemm_heuristic.sm90_candidates(desc))
    assert ranked is not None

    with pytest.warns(RuntimeWarning, match="analytical fallback"):
        res = epi_autotune.tuned_mod_gemm(
            mod,
            A,
            B,
            D,
            epi_args={"stat": buf},
            b_kn=True,
            cu_seqlens_m=cu,
            tune_mode=TuneMode.STATIC,
        )

    assert res.config is None  # fallback provenance, unchanged
    assert res.plan == "plan-sentinel"
    assert set(res.sinks) == {"stat"}  # was {} -> KeyError in the eager frontend
    expected = (m, -(-n // ranked.tile_n))
    assert tuple(res.sinks["stat"].shape) == expected
    assert res.sinks["stat"].data_ptr() == buf.data_ptr()  # live view of the buffer
    # The delegated call ran the ranked tiles and consumed the same slice.
    ((delegated_kwargs, delegated_epi),) = mod.gemm_calls
    assert delegated_kwargs["tile_N"] == ranked.tile_n
    assert delegated_kwargs["tile_M"] == ranked.tile_m
    assert tuple(delegated_epi["stat"].shape) == expected
    # The stash the hand-off read: the ranked config itself, views identical.
    tuner = next(iter(epi_autotune._MOD_TUNERS.values()))
    stash_cfg, stash_sinks = tuner.fn._last_fallback
    assert stash_cfg == ranked
    assert stash_sinks["stat"] is res.sinks["stat"]


def test_fallback_default_config_path_stashes_sliced_sinks():
    """Uncovered family WITH sinks (dense): the -1 chain resolves the mod's
    own default config and takes the explicit-tile path -- the stash reports
    that config's sliced views, and the delegation carries tile parameters."""
    import torch

    import cute_gemm.gemm_runtime.autotune as epi_autotune

    mod = _stub_sink_mod("default_cfg_sink_mod")
    fn = epi_autotune._make_tuned_fn(mod, ("stat",))
    buf = torch.full((64, 2), 3.0)
    out = fn(
        A=torch.empty((64, 16), dtype=torch.bfloat16),
        B=torch.empty((128, 16), dtype=torch.bfloat16),
        D=torch.empty((64, 128), dtype=torch.bfloat16),
        stat=buf,
    )

    assert out == "plan-sentinel"
    cfg, sinks = fn._last_fallback
    assert cfg == mod._default_config(None, None, None)
    assert tuple(sinks["stat"].shape) == (64, 1)  # ceil(n=128 / tile_n=128)
    ((delegated_kwargs, delegated_epi),) = mod.gemm_calls
    assert delegated_kwargs["tile_N"] == 128  # explicit tiles, not the blind call
    assert delegated_epi["stat"] is sinks["stat"]


def test_fallback_bare_path_stash_covers_sinkless_mods():
    """A sink-less uncovered mod still rides the blind low-level delegation;
    the stash is still written (empty), so tuned_mod_gemm's hand-off read
    never faces a missing attribute."""
    import torch

    import cute_gemm.gemm_runtime.autotune as epi_autotune

    record = []

    class _BareMod:
        _ident = "bare_mod"

        def __init__(self):
            self.sinks = {}

        def gemm(self, A, B, D, C=None, *, epi_args, **kwargs):
            record.append(kwargs)
            return "resolved"

    fn = epi_autotune._make_tuned_fn(_BareMod(), ())
    out = fn(
        A=torch.empty((64, 16), dtype=torch.bfloat16),
        B=torch.empty((128, 16), dtype=torch.bfloat16),
        D=torch.empty((64, 128), dtype=torch.bfloat16),
    )

    assert out == "resolved"
    assert fn._last_fallback == (None, {})
    assert "tile_M" not in record[0]  # loop-free low-level delegation


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="fallback lifecycle needs a GPU"
)
def test_varlen_sink_running_fallback_returns_sliced_sinks(tmp_path, monkeypatch):
    """Full eager lifecycle of the fixed KeyError: RUNNING, same contract F,
    two cold winner buckets W.  Bucket 1 sweeps and consumes its winner;
    bucket 2 spends F's quota and REUSES the contract winner (2026-10-22) --
    an explicit config whose sliced sink views must still come back.  Both
    buckets are checked against a torch reference."""
    from itertools import pairwise

    import torch

    import cute_gemm.gemm_runtime.autotune as epi_autotune
    from cute_gemm.epilogue.frontend import gemm_epilogue
    from cute_gemm.epilogue.ops import ColVecReduce
    from cute_gemm.gemm_config import TuneMode

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))  # hermetic winner cache
    monkeypatch.setattr(epi_autotune, "_MOD_TUNERS", {})

    @gemm_epilogue(reduces={"stat": ColVecReduce("stat")})
    def _stat_rowsum_epi(acc):
        return {"D": acc, "stat": acc}

    device = "cuda"
    if torch.cuda.get_device_capability(device)[0] != 9:
        pytest.skip("the analytical -1 fallback ranks SM90 only")
    torch.random.manual_seed(7)
    experts, k, n = 8, 256, 512
    B = torch.randn((experts, k, n), device=device, dtype=torch.bfloat16) / 8

    def run(rows):
        # avg rows/expert 128 vs 256: two W buckets, one contract F (the
        # ragged extent collapses to a mark; E/K/N/dtype are what is left).
        A = torch.randn((rows, k), device=device, dtype=torch.bfloat16) / 8
        cu = torch.arange(0, rows + 1, rows // experts, device=device, dtype=torch.int32)
        res = _stat_rowsum_epi(A, B, tune_mode=TuneMode.RUNNING, cu_seqlens_m=cu)
        # B rides the torch convention (E, K, N): each expert's segment is a
        # plain A_seg @ B[e].
        ref = torch.cat([A[s:e].float() @ B[i].float() for i, (s, e) in enumerate(pairwise(cu))])
        return res, ref

    res1, ref1 = run(1024)
    tuners = list(epi_autotune._MOD_TUNERS.values())
    assert len(tuners) == 1 and tuners[0].used_fallback is False  # bucket 1 swept
    assert res1["stat"].shape == (1024,)
    assert (res1["D"].float() - ref1).abs().max() < 2e-2 * ref1.abs().max()
    assert (res1["stat"] - ref1.sum(-1)).abs().max() < 2e-2 * ref1.abs().max()

    res2, ref2 = run(2048)
    # 2026-10-22: quota spent reuses the contract's measured winner (an
    # explicit config) -- NOT the -1 fallback.  Sinks come back sliced for
    # that config, which is what closed the original KeyError.
    assert tuners[0].used_fallback is False
    assert tuners[0].best_config is not None
    assert res2["stat"].shape == (2048,)  # sinks actually returned (the KeyError path)
    assert (res2["D"].float() - ref2).abs().max() < 2e-2 * ref2.abs().max()
    assert (res2["stat"] - ref2.sum(-1)).abs().max() < 2e-2 * ref2.abs().max()


def test_concurrent_contract_winner_recording_is_atomic(_cpu_tuning, tmp_path):
    """The 2026-10-27 incident, CPU edition.

    torchrun ranks share one contract_winners file. Concurrent recorders of
    the same contract must neither crash on a peer's in-flight write
    (JSONDecodeError escaped _record_contract_winner and killed every later
    run of the affected contracts) nor leave torn JSON behind. Also pins the
    single-resolution-point rule: the winner file must land inside this
    test's DLKERNEL_CACHE_DIR, never in a shared home directory.
    """
    import json
    import multiprocessing as mp

    import cute_gemm.autotuner as at

    contract = ("contract-race",)
    n_proc, n_iter = 4, 800
    cfgs = (at.AutotuneConfig(block=1), at.AutotuneConfig(block=123456))

    def make_tuner():
        return at.Autotuner(
            lambda x, block=1: None,
            configs=list(cfgs),
            key=[],
            do_bench=lambda fn, **k: [1.0],
            cache_results=True,
        )

    def hammer(rank, q):
        tuner = make_tuner()
        cfg = cfgs[rank % 2]  # drifted candidate spaces: str() lengths differ
        n_err, first = 0, None
        for _ in range(n_iter):
            try:
                tuner._record_contract_winner(contract, cfg)
            except ValueError as e:  # JSONDecodeError subclasses ValueError
                n_err, first = n_err + 1, first or str(e)
        q.put((n_err, first))

    def read_through(path, stop, q):
        n_bad, first = 0, None
        while not stop.is_set():
            try:
                json.loads(path.read_text())
            except FileNotFoundError:
                continue
            except ValueError as e:
                n_bad, first = n_bad + 1, first or str(e)
        q.put((n_bad, first))

    path = make_tuner()._contract_winner_path(contract)
    assert path.is_relative_to(tmp_path), "contract winners bypassed the cache root"

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    stop = ctx.Event()
    reader = ctx.Process(target=read_through, args=(path, stop, q))
    procs = [ctx.Process(target=hammer, args=(r, q)) for r in range(n_proc)]
    reader.start()
    for p in procs:
        p.start()
    results = [q.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(60)
    stop.set()
    results.append(q.get(timeout=60))
    reader.join(60)

    for n_err, first in results:
        assert n_err == 0, f"recorder crashed / reader saw torn JSON: {first}"
    json.loads(path.read_text())  # a torn terminal state fails here
