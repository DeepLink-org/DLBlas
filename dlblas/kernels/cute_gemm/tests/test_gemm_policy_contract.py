"""Contract tests for the staged GEMM tuning policy."""

import pytest
import torch

from cute_gemm.autotuner import AutotuneConfig
from cute_gemm.gemm_epilogue_tune_policy import GemmEpilogueTunePolicy
from cute_gemm.gemm_tune_policy import GemmTunePolicy
from cute_gemm.varlen_selector import VarlenGemmTunePolicy


def test_gemm_policy_binds_signature_and_stages_selection(monkeypatch):
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    events = []

    def structural(configs, named_args, **kwargs):
        events.append(("structural", tuple(configs), dict(named_args), dict(kwargs)))
        return list(configs[1:])

    def selector(configs, **kwargs):
        events.append(("selector", tuple(configs), dict(kwargs)))
        return list(configs[:1])

    monkeypatch.setattr("cute_gemm.gemm_tune_policy.prune_structural_gemm_configs", structural)
    monkeypatch.setattr("cute_gemm.varlen_selector.select_varlen_gemm_candidates", selector)
    parent_shortlist_calls = []
    parent_shortlist = GemmTunePolicy.shortlist

    def record_parent_shortlist(self, configs, named_args, kwargs):
        parent_shortlist_calls.append(tuple(configs))
        return parent_shortlist(self, configs, named_args, kwargs)

    monkeypatch.setattr(GemmTunePolicy, "shortlist", record_parent_shortlist)
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(
        ("A", "B", "out", "cu_seqlens_m"),
        (None, None, None, None),
    )
    A = torch.empty((64, 16), dtype=torch.bfloat16)
    B = torch.empty((16, 32), dtype=torch.bfloat16)
    out = torch.empty((64, 32), dtype=torch.bfloat16)
    cu = torch.empty((3,), dtype=torch.int32)
    configs = [AutotuneConfig(config="a"), AutotuneConfig(config="b")]

    key = policy.make_key((A, B, out, cu), {}, lambda: ("default",))
    assert key[0] == "gemm_token_bucket_v6"
    prepared = policy.prepare_candidates(configs, {"A": A}, {"cu_seqlens_m": cu})
    selected = policy.shortlist(prepared, {"A": A}, {"cu_seqlens_m": cu})

    assert [event[0] for event in events] == ["structural", "selector"]
    assert parent_shortlist_calls == [(AutotuneConfig(config="b"),)]
    assert selected == prepared[:1]


def test_gemm_policy_keeps_dense_calls_on_default_key_and_skips_selector():
    policy = GemmTunePolicy()
    policy.bind_signature(("A", "B"), (None, None))
    A = torch.empty((8, 4), dtype=torch.float16)
    B = torch.empty((4, 8), dtype=torch.float16)
    configs = [AutotuneConfig(config="a")]

    assert policy.make_key((A, B), {}, lambda: ("default",)) == ("default",)
    assert policy.shortlist(configs, {"A": A, "B": B}, {}) == configs


def test_gemm_policy_budget_override_tracks_the_bucket_key():
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(("A", "B", "out", "cu_seqlens_m"), (None,) * 4)
    A = torch.empty((64, 16), dtype=torch.bfloat16)
    B = torch.empty((3, 16, 32), dtype=torch.bfloat16)
    out = torch.empty((64, 32), dtype=torch.bfloat16)
    cu = torch.empty((3,), dtype=torch.int32)

    varlen_key = policy.make_key((A, B, out, cu), {}, lambda: ("default",))
    assert policy.budget_override(varlen_key) is True

    dense_policy = GemmTunePolicy()
    dense_policy.bind_signature(("A", "B"), (None, None))
    dense_key = dense_policy.make_key((A, B), {}, lambda: ("default",))
    assert dense_key == ("default",)
    assert dense_policy.budget_override(dense_key) is False


# --------------------------------------------------------------------------- #
# Varlen bucket key semantics (CPU, DLKERNEL_ARCH=90 proxy)
# --------------------------------------------------------------------------- #


def _m_group_workload(total_m, experts=32, n=512, k=512):
    A = torch.empty((total_m, k), dtype=torch.bfloat16)
    B = torch.empty((experts, n, k), dtype=torch.bfloat16)
    out = torch.empty((total_m, n), dtype=torch.bfloat16)
    cu = torch.empty((experts + 1,), dtype=torch.int32)
    return A, B, out, cu


def test_same_bucket_same_leaf_shares_key_and_cross_leaf_splits(monkeypatch):
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(("A", "B", "out", "cu_seqlens_m"), (None,) * 4)

    # avg/expert 1100 / 1200 / 1250 all round to the same 2048 bucket, but the
    # M-group leaf boundary (1194.67) splits the first from the other two.
    key_low = policy.make_key((*_m_group_workload(32 * 1100),), {}, lambda: ("d",))
    key_high_1 = policy.make_key((*_m_group_workload(32 * 1200),), {}, lambda: ("d",))
    key_high_2 = policy.make_key((*_m_group_workload(32 * 1250),), {}, lambda: ("d",))

    assert key_high_1 == key_high_2  # same bucket + same leaf -> shared winner
    assert key_low != key_high_1  # same bucket, different leaf -> tuned apart
    assert ("avg_expert_m", 2048) in key_low and ("avg_expert_m", 2048) in key_high_1


def test_epilogue_policy_buckets_routed_mcolvec_broadcast(monkeypatch):
    """The routed score vector shares the same M bucket as the expert rows."""
    from types import SimpleNamespace

    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    policy = GemmEpilogueTunePolicy(SimpleNamespace(outputs=(), sinks={}))
    policy.bind_signature(
        ("A", "B", "D", "mColVecBroadcast", "cu_seqlens_m"),
        (None,) * 5,
    )

    def key_for(total_m):
        A = torch.empty((total_m, 512), dtype=torch.bfloat16)
        B = torch.empty((32, 512, 512), dtype=torch.bfloat16)
        D = torch.empty((total_m, 512), dtype=torch.bfloat16)
        route = torch.empty((total_m,), dtype=torch.bfloat16)
        cu = torch.empty((33,), dtype=torch.int32)
        return policy.make_key(
            (),
            {"A": A, "B": B, "D": D, "mColVecBroadcast": route, "cu_seqlens_m": cu},
            lambda: ("dense",),
        )

    # Both workloads are in the same avg-M bucket and the same m-high selector leaf.
    assert key_for(32 * 1200) == key_for(32 * 1250)
    # The open tail shares one average bucket; the 1024 boundary still splits.
    assert key_for(32 * 2500) == key_for(32 * 1250)
    assert key_for(32 * 1024) != key_for(32 * 1024 + 1)

    dense = torch.empty((32 * 1250,), dtype=torch.bfloat16)
    dense_key = policy.make_key(
        (),
        {
            "A": torch.empty((32 * 1250, 512), dtype=torch.bfloat16),
            "B": torch.empty((32, 512, 512), dtype=torch.bfloat16),
            "D": torch.empty((32 * 1250, 512), dtype=torch.bfloat16),
            "mColVecBroadcast": dense,
            "cu_seqlens_m": None,
        },
        lambda: ("dense",),
    )
    assert dense_key == ("dense",)


def test_epilogue_policy_buckets_scratch_and_d_output(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    mod = SimpleNamespace(
        outputs=("premult",),
        sinks={"sqsum": SimpleNamespace(sink_alloc_shape=lambda *a, **k: None)},
    )
    policy = GemmEpilogueTunePolicy(mod)
    arg_names = (
        "A",
        "B",
        "D",
        "C",
        "mod_digest",
        "b_kn",
        "cu_seqlens_m",
        "A_idx",
        "dynamic_scheduler",
        "SFA",
        "SFB",
        "bs_format_a",
        "bs_format_b",
        "concat_layout",
        "transform_digest",
        "transform_sf",
        "config",
        "ta__scale",
        "sqsum",
        "premult",
    )
    policy.bind_signature(arg_names, (None,) * len(arg_names))

    def call(total_m):
        A = torch.empty((total_m, 512), dtype=torch.bfloat16)
        B = torch.empty((32, 512, 512), dtype=torch.bfloat16)
        D = torch.empty((total_m, 512), dtype=torch.bfloat16)  # ragged rows
        cu = torch.empty((33,), dtype=torch.int32)
        # Sink scratch + transform operand: shapes track total_m on purpose --
        # they must NOT split the bucket (E1: dtype-keyed only).  premult is a
        # per-row epi OUTPUT: its row dim equals the ragged M and is caught by
        # the value-matched rewrite (neither sink nor ta__).
        sqsum = torch.empty((32, total_m, 4), dtype=torch.float32)
        ta_scale = torch.empty((total_m,), dtype=torch.float32)
        premult = torch.empty((total_m, 512), dtype=torch.bfloat16)
        kwargs = dict(
            A=A,
            B=B,
            D=D,
            mod_digest="m1",
            b_kn=False,
            cu_seqlens_m=cu,
            dynamic_scheduler=False,
            concat_layout=None,
            transform_digest=None,
            sqsum=sqsum,
            premult=premult,
            ta__scale=ta_scale,
        )
        return policy.make_key((), kwargs, lambda: ("dense-default",)), kwargs

    key_1, _ = call(32 * 1200)
    key_2, _ = call(32 * 1250)  # same bucket: D/sink/ta shapes differ, keys equal
    assert key_1 == key_2
    assert key_1[0] != "dense-default"
    assert policy.budget_override(key_1) is True

    key_3, _ = call(32 * 8)  # different bucket (avg 8 -> floor 32)
    assert key_3 != key_1

    dense_kwargs = dict(call(256)[1], cu_seqlens_m=None)
    dense_key = policy.make_key((), dense_kwargs, lambda: ("dense-default",))
    assert dense_key == ("dense-default",)
    assert policy.budget_override(dense_key) is False


def test_epilogue_and_plain_policies_share_varlen_key_facts(monkeypatch):
    """Same logical varlen workload keys on identical facts through both
    entries (bucket, ragged-rewritten shapes, selector leaf)."""
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    from types import SimpleNamespace

    from cute_gemm.gemm_epilogue_tune_policy import GemmEpilogueTunePolicy

    plain = VarlenGemmTunePolicy()
    plain.bind_signature(("A", "B", "out", "cu_seqlens_m"), (None,) * 4)
    mod = SimpleNamespace(outputs=(), sinks={})
    epi = GemmEpilogueTunePolicy(mod)
    epi.bind_signature(("A", "B", "D", "cu_seqlens_m"), (None,) * 4)

    total_m, experts, n, k = 32 * 1200, 32, 512, 512
    A = torch.empty((total_m, k), dtype=torch.bfloat16)
    B = torch.empty((experts, n, k), dtype=torch.bfloat16)
    out = torch.empty((total_m, n), dtype=torch.bfloat16)
    D = torch.empty((total_m, n), dtype=torch.bfloat16)
    cu = torch.empty((experts + 1,), dtype=torch.int32)

    plain_key = plain.make_key((A, B, out, cu), {}, lambda: ("d",))
    epi_key = epi.make_key((), dict(A=A, B=B, D=D, cu_seqlens_m=cu), lambda: ("d",))

    def facts(key):
        # The key interleaves scalar tuples with bare tensor runs (name,
        # shape, stride, dtype); compare only the shared semantic facts.
        wanted = {"avg_expert_m", "selector_leaf"}
        semantic = [entry for entry in key if isinstance(entry, tuple) and entry[0] in wanted]
        i = key.index("A")
        return semantic, key[i : i + 4]

    assert facts(plain_key) == facts(epi_key)


def test_signature_missing_varlen_params_is_dense(monkeypatch):
    """Regression: a signature without cu_seqlens_k used to bind the _MISSING
    sentinel, which read as "varlen K" and bucket-keyed every dense call
    through such a signature (the epilogue tuner's)."""
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    policy = VarlenGemmTunePolicy()
    # cu_seqlens_m present (None), cu_seqlens_k absent from the signature.
    policy.bind_signature(("A", "B", "out", "cu_seqlens_m"), (None,) * 4)
    A = torch.empty((64, 32), dtype=torch.bfloat16)
    B = torch.empty((16, 32), dtype=torch.bfloat16)
    out = torch.empty((64, 16), dtype=torch.bfloat16)

    key = policy.make_key((A, B, out), {"cu_seqlens_m": None}, lambda: ("dense",))
    assert key == ("dense",)
    assert policy.budget_override(key) is False


def test_varlen_policy_is_the_operator_specific_gemm_subclass():
    from cute_gemm.gemm_tune_policy import GemmTunePolicy
    from cute_gemm.varlen_selector import VarlenGemmTunePolicy

    assert issubclass(VarlenGemmTunePolicy, GemmTunePolicy)
    assert VarlenGemmTunePolicy.__module__ == "cute_gemm.varlen_selector"


def test_varlen_policy_delegates_dense_key_and_shortlist_to_parent():
    policy = VarlenGemmTunePolicy()
    policy.bind_signature(("A", "B"), (None, None))
    configs = [AutotuneConfig(config="a"), AutotuneConfig(config="b")]
    A = torch.empty((8, 4), dtype=torch.float16)
    B = torch.empty((4, 8), dtype=torch.float16)
    assert policy.make_key((A, B), {}, lambda: ("dense",)) == ("dense",)
    assert policy.shortlist(configs, {"A": A, "B": B}, {}) == configs


def test_epilogue_policy_uses_explicit_ragged_roles(monkeypatch):
    from types import SimpleNamespace

    from cute_gemm.gemm_epilogue_tune_policy import GemmEpilogueTunePolicy

    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    mod = SimpleNamespace(outputs=("premult",), sinks={})
    policy = GemmEpilogueTunePolicy(mod)
    policy.bind_signature(("A", "B", "D", "cu_seqlens_m", "accidental"), (None,) * 5)

    def key_for(total_m):
        A = torch.empty((total_m, 512), dtype=torch.bfloat16)
        B = torch.empty((32, 512, 512), dtype=torch.bfloat16)
        D = torch.empty((total_m, 512), dtype=torch.bfloat16)
        accidental = torch.empty((total_m, 7), dtype=torch.float32)
        cu = torch.empty((33,), dtype=torch.int32)
        return policy.make_key(
            (),
            {"A": A, "B": B, "D": D, "cu_seqlens_m": cu, "accidental": accidental},
            lambda: ("dense",),
        )

    # The declared output is bucketed; an unrelated tensor that merely happens
    # to have the same M extent remains exact and therefore separates keys.
    assert key_for(32 * 1200) != key_for(32 * 1250)


def test_policy_signature_binding_is_immutable():
    policy = GemmTunePolicy()
    policy.bind_signature(("A", "B"), (None, None))
    policy.bind_signature(("A", "B"), (None, None))
    with pytest.raises(ValueError, match="already bound"):
        policy.bind_signature(("A", "B", "out"), (None, None, None))


def test_varlen_row_reduce_combine_rule_is_shared_with_prune():
    from types import SimpleNamespace

    from cute_gemm.epilogue.legality import prune_epilogue_configs
    from cute_gemm.gemm_config import GemmConfig

    class RowSink:
        dim = 1
        check_oob = True
        combine = "max"

    mod = SimpleNamespace(mode="element", outputs=(), sinks={"stats": RowSink()})
    config = AutotuneConfig(
        config=GemmConfig(
            tile_m=128,
            tile_n=128,
            cluster_m=1,
            cluster_n=1,
            device_capacity=9,
            is_dynamic_persistent=False,
        )
    )
    named = {
        "A": torch.empty((128, 16), dtype=torch.bfloat16),
        "B": torch.empty((16, 128), dtype=torch.bfloat16),
        "cu_seqlens_m": torch.empty((2,), dtype=torch.int32),
    }
    assert prune_epilogue_configs(mod, None, [config], named, device_capacity=(9, 0)) == []
