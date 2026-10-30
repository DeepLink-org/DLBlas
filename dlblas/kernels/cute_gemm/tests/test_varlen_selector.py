"""Tests for the varlen grouped-GEMM candidate selector.

The selector is two plain rule trees: a two-leaf K-group tree on ``K / N`` and
a two-leaf M-group tree on ``avg_m``.  These self-contained tests pin:

* the frozen constants really are the tree the offline study validated;
* the selector only ever *filters* - never invents a candidate, never returns an
  empty set, obeys the 8-candidate budget when applied, never depends on the input
  order, and falls back to the full legal set outside the measured envelope;
* the exact four leaf candidate sets and threshold/coverage boundaries.

Measured hit rates remain offline research evidence, not a CI test whose
private dataset can silently disappear from a clean checkout.
"""

from __future__ import annotations

import pytest

from cute_gemm.gemm_config import _get_sm90_configs, get_sm90_dynamic_varlen_configs
from cute_gemm.varlen_selector import (
    FALLBACK_LEAF_ID,
    KEEP,
    K_GROUP_TREE,
    MIN_RULE_KEEP,
    M_GROUP_TREE,
    SELECTOR_VERSION,
    TRAINED_ENVELOPE,
    candidate_fields,
    canonical_filler_key,
    rule_table,
    select_varlen_gemm_candidates,
    selector_leaf_id,
    workload_metadata,
)

#: the gap the deployed K-group cut sits in
K_GROUP_SPLIT_GAP = (78.14453125, 80.0)

K_GROUP_LOW_RULE, K_GROUP_HIGH_RULE = K_GROUP_TREE[0], K_GROUP_TREE[-1]
K_GROUP_SPLIT_THRESHOLD = K_GROUP_LOW_RULE["upper"]


class _Tensor:
    """Shape/stride/dtype stand-in; the selector only reads those."""

    def __init__(self, shape, stride, dtype="torch.bfloat16"):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self.dtype = dtype

    def stride(self):
        return self._stride


def _cu_seqlens(expert_count: int, routed_rows: int):
    """Monotone cu_seqlens with a known expert count that sums to routed_rows."""
    base, extra = divmod(int(routed_rows), int(expert_count))
    offsets = [0]
    for index in range(expert_count):
        offsets.append(offsets[-1] + base + (1 if index < extra else 0))
    return offsets


def _pool(group: str):
    """The structurally legal SM90 candidate set for a group, as prune leaves it.

    Mirrors ``gemm_interface.prune_invalid_gemm_configs``: the static grid plus
    the dynamic-persistent twins for varlen_k only, then the structural
    varlen_m swap filter.
    """
    configs = _get_sm90_configs()
    if group == "K-group":  # varlen_k gains the dynamic twins
        configs = configs + get_sm90_dynamic_varlen_configs()
    if group == "M-group":  # varlen_m prunes swap_ab
        configs = [config for config in configs if not config.swap_ab]
    return configs


def _ids(kept):
    return {
        (
            candidate_fields(config)["tile_m"],
            candidate_fields(config)["tile_n"],
            candidate_fields(config)["cluster_m"],
            candidate_fields(config)["cluster_n"],
            candidate_fields(config)["swap_ab"],
            candidate_fields(config)["is_dynamic_persistent"],
        )
        for config in kept
    }


def _select(group: str, **workload):
    workload.setdefault("device_capability", 9)
    workload.setdefault("dtype", "torch.bfloat16")
    return select_varlen_gemm_candidates(_pool(group), **workload)


def _select_from(pool, **workload):
    workload.setdefault("device_capability", 9)
    workload.setdefault("dtype", "torch.bfloat16")
    return select_varlen_gemm_candidates(pool, **workload)


def _m_workload(routed_rows: int = 4096, experts: int = 64, N: int = 1536, K: int = 2048):
    return {
        "A": _Tensor((routed_rows, K), (1, routed_rows)),
        "B": _Tensor((K, N), (N, 1)),
        "cu_seqlens_m": _cu_seqlens(experts, routed_rows),
    }


def _k_workload(routed_rows: int = 8192, experts: int = 64, N: int = 2048):
    return {
        "A": _Tensor((2048, routed_rows), (1, 2048)),
        "B": _Tensor((routed_rows, N), (N, 1)),
        "cu_seqlens_k": _cu_seqlens(experts, routed_rows),
    }


def test_rule_table_is_the_validated_tree():
    table = rule_table()
    assert table["selector_version"] == SELECTOR_VERSION
    assert table["keep"] == KEEP == 8
    assert table["device_capability"] == 9
    assert table["dtype"] == "bfloat16"
    assert table["split_feature"] == "K_over_N"
    tree = table["k_tree"]
    assert len(tree) == 2, "the studied tree has two leaves"
    low, high = tree
    gap_low, gap_high = K_GROUP_SPLIT_GAP
    assert gap_low < low["upper"] < gap_high, "the deployed cut must sit in the measured gap"
    assert low["swap_ab"] is True and sorted(low["clusters"]) == ["1,2", "2,1"]
    assert high["upper"] == float("inf") and high["clusters"] == ["1,2"]
    assert high["swap_ab"] is None
    # v7: only k-high admits the measured dynamic-persistent twins
    assert high["dynamic_shapes"] == ["256x128"]
    m_tree = table["m_tree"]
    assert all("dynamic_shapes" not in leaf for leaf in tree[:1] + m_tree)
    assert len(m_tree) == 2, "the M-group tree has two leaves"
    assert m_tree[0]["upper"] < m_tree[1]["upper"] == float("inf")
    assert table["m_split_feature"] == "avg_m"
    for leaf in m_tree:
        assert leaf["swap_ab"] is False and leaf["clusters"] == ["1,2"]
    assert MIN_RULE_KEEP > 0
    for group, envelope in TRAINED_ENVELOPE.items():
        assert envelope["N"][0] < envelope["N"][1] and group.endswith("group")


def test_m_group_tree_switches_leaf_at_the_studied_average():
    small = _m_workload(routed_rows=64 * 100, experts=64)  # avg_m = 100
    large = _m_workload(routed_rows=64 * 4096, experts=64)  # avg_m = 4096
    low, high = _select("M-group", **small), _select("M-group", **large)
    assert len(low) == len(high) == KEEP
    assert low != high
    assert (128, 160, 1, 2, False, False) in _ids(low) and (256, 128, 1, 2, False, False) in _ids(
        low
    )
    assert (128, 192, 1, 2, False, False) in _ids(high) and (256, 208, 1, 2, False, False) in _ids(
        high
    )
    assert (128, 160, 1, 2, False, False) not in _ids(high)
    assert (256, 208, 1, 2, False, False) not in _ids(low)
    # no leaf of the M-group tree ever admits a dynamic candidate
    assert not any(entry[-1] for entry in _ids(low) | _ids(high))


def test_k_group_tree_switches_leaf_at_the_studied_ratio():
    low = _select("K-group", **_k_workload(routed_rows=8192, N=2048))
    high = _select("K-group", **_k_workload(routed_rows=262144, N=2048))
    assert len(low) == len(high) == KEEP
    assert low != high
    # 8192/2048 = 4 < 79.07 < 128 = 262144/2048
    assert (128, 256, 1, 2, True, False) in _ids(low) and (256, 192, 2, 1, True, False) in _ids(low)
    assert (192, 128, 1, 2, False, False) in _ids(high) and (256, 128, 1, 2, False, False) in _ids(
        high
    )
    assert (128, 256, 1, 2, True, False) not in _ids(high)
    assert (192, 128, 1, 2, False, False) not in _ids(low)
    # the dynamic twins live on k-high only
    assert (256, 128, 1, 2, False, True) in _ids(high) and (256, 128, 1, 2, True, True) in _ids(
        high
    )
    assert not any(entry[-1] for entry in _ids(low))


@pytest.mark.parametrize(
    "group,workload,expected",
    [
        (
            "K-group",
            _k_workload(),
            {
                (128, 256, 1, 2, True, False),
                (128, 256, 2, 1, True, False),
                (256, 192, 1, 2, True, False),
                (256, 192, 2, 1, True, False),
                (128, 128, 1, 2, False, False),
                (128, 128, 1, 2, True, False),
                (128, 128, 2, 1, False, False),
                (128, 128, 2, 1, True, False),
            },
        ),
        (
            "K-group",
            _k_workload(routed_rows=262144),
            {
                (192, 128, 1, 2, False, False),
                (192, 128, 1, 2, True, False),
                (256, 128, 1, 2, False, False),
                (256, 128, 1, 2, True, False),
                (256, 192, 1, 2, False, False),
                (256, 192, 1, 2, True, False),
                # v7: the measured dynamic-persistent twins join the k-high body
                (256, 128, 1, 2, False, True),
                (256, 128, 1, 2, True, True),
            },
        ),
        (
            "M-group",
            _m_workload(),
            {
                (128, 160, 1, 2, False, False),
                (128, 192, 1, 2, False, False),
                (128, 208, 1, 2, False, False),
                (128, 224, 1, 2, False, False),
                (128, 256, 1, 2, False, False),
                (192, 128, 1, 2, False, False),
                (256, 128, 1, 2, False, False),
                (128, 128, 1, 2, False, False),
            },
        ),
        (
            "M-group",
            _m_workload(routed_rows=64 * 4096),
            {
                (128, 192, 1, 2, False, False),
                (128, 256, 1, 2, False, False),
                (192, 128, 1, 2, False, False),
                (256, 128, 1, 2, False, False),
                (256, 160, 1, 2, False, False),
                (256, 192, 1, 2, False, False),
                (256, 208, 1, 2, False, False),
                (128, 128, 1, 2, False, False),
            },
        ),
    ],
)
def test_four_leaves_return_the_frozen_candidate_sets(group, workload, expected):
    kept = _select(group, **workload)
    assert len(kept) == KEEP
    assert _ids(kept) == expected


def test_k_high_dynamic_twins_come_after_the_static_body():
    """The v7 twins fill the k-high budget exactly, after the 6 static matches,
    and the twin count leaves no room for canonical fillers."""
    from cute_gemm.gemm_config import GemmConfig

    high = _select("K-group", **_k_workload(routed_rows=262144, N=2048))
    ids = [
        (
            candidate_fields(config)["is_dynamic_persistent"],
            candidate_fields(config)["tile_m"],
            candidate_fields(config)["swap_ab"],
        )
        for config in high
    ]
    assert ids[-2:] == [(True, 256, False), (True, 256, True)], (
        "the dynamic twins close the list, in canonical order (swap False then True)"
    )
    assert all(not dynamic for dynamic, *_ in ids[:-2])

    # without the twins in the pool (static-only), the studied pre-v7 set is
    # reproduced exactly: 6 static matches padded with 128x128 fillers
    static_pool = _get_sm90_configs()
    kept = _select_from(static_pool, **_k_workload(routed_rows=262144, N=2048))
    assert len(kept) == KEEP
    assert not any(candidate_fields(config)["is_dynamic_persistent"] for config in kept)
    filler = {
        (candidate_fields(config)["tile_m"], candidate_fields(config)["tile_n"]) for config in kept
    } - {(192, 128), (256, 128), (256, 192)}
    assert filler == {(128, 128)}

    # a dynamic candidate a rule does not admit is rejected even when handed in:
    # the stranger's static fields match k-low's rule exactly, so only the
    # dynamic guard can keep it out of the body
    stranger = GemmConfig(
        tile_m=128,
        tile_n=256,
        pingpong=False,
        cluster_m=1,
        cluster_n=2,
        swap_ab=True,
        device_capacity=9,
        is_dynamic_persistent=True,
    )
    kept = _select_from(static_pool + [stranger], **_k_workload(routed_rows=8192, N=2048))
    assert all(not candidate_fields(config)["is_dynamic_persistent"] for config in kept)


@pytest.mark.parametrize(
    "workload,expected_leaf",
    [
        (_k_workload(routed_rows=161940), "k-low"),
        (_k_workload(routed_rows=161941), "k-high"),
        (_m_workload(routed_rows=35840, experts=30), "fallback"),
        (_m_workload(routed_rows=71680, experts=60), "m-low"),
        (_m_workload(routed_rows=71681, experts=60), "m-high"),
    ],
)
def test_leaf_boundaries_and_coverage_are_self_contained(workload, expected_leaf):
    assert (
        selector_leaf_id(**workload, dtype="torch.bfloat16", device_capability=9) == expected_leaf
    )


def test_k_group_low_leaf_is_padded_with_the_canonical_order():
    low = _select("K-group", **_k_workload(routed_rows=8192, N=2048))
    body = {
        (tile_m, tile_n, cluster_m, cluster_n, swap_ab, False)
        for tile_m, tile_n in K_GROUP_LOW_RULE["shapes"]
        for cluster_m, cluster_n in K_GROUP_LOW_RULE["clusters"]
        for swap_ab in (K_GROUP_LOW_RULE["swap_ab"],)
    }
    filler = _ids(low) - body
    assert len(body) == 4 and filler, "the low leaf body is smaller than the budget"
    assert {entry[:2] for entry in filler} == {(128, 128)}, (
        "``canonical_filler_key`` starts at the smallest tile area, so 128x128 fills first"
    )


def test_selector_never_invents_candidates_and_never_empties_the_set():
    for group, workload in (("K-group", _k_workload()), ("M-group", _m_workload())):
        pool = _pool(group)
        kept = _select_from(pool, **workload)
        assert 0 < len(kept) <= KEEP < len(pool)
        assert all(any(kept_config is config for config in pool) for kept_config in kept)


def test_result_is_independent_of_input_order():
    pool = _pool("K-group")
    workload = _k_workload()
    reference_ids = sorted(id(config) for config in _select_from(pool, **workload))
    again = _select_from(pool, **workload)
    assert reference_ids == sorted(id(config) for config in again)
    reordered = _select_from(list(reversed(pool)), **workload)
    assert reference_ids == sorted(id(config) for config in reordered)
    assert [id(config) for config in again] == [id(config) for config in reordered], (
        "the returned order must not depend on the caller's candidate order"
    )
    # the body (4 candidates for the low leaf) comes first, then the padding,
    # and the padding itself is in canonical order
    keys = [canonical_filler_key(candidate_fields(config)) for config in again]
    body_size = len(K_GROUP_LOW_RULE["shapes"]) * len(K_GROUP_LOW_RULE["clusters"])
    assert body_size == 4, "the low leaf body is 2 shapes x 2 clusters x swap=True"
    assert keys[body_size:] == sorted(keys[body_size:])


def test_dense_and_uncovered_layouts_return_the_pool_unchanged():
    pool = _pool("M-group")
    dense = select_varlen_gemm_candidates(
        pool,
        A=_Tensor((4096, 2048), (1, 4096)),
        B=_Tensor((2048, 1536), (1536, 1)),
        device_capability=9,
        dtype="torch.bfloat16",
    )
    assert len(dense) == len(pool)
    gather = select_varlen_gemm_candidates(
        pool,
        A=_Tensor((4096, 2048), (1, 4096)),
        B=_Tensor((2048, 1536), (1536, 1)),
        cu_seqlens_m=_cu_seqlens(64, 4096),
        A_idx=[0],
        device_capability=9,
        dtype="torch.bfloat16",
    )
    assert len(gather) == len(pool)
    both = select_varlen_gemm_candidates(
        pool,
        A=_Tensor((4096, 2048), (1, 4096)),
        B=_Tensor((2048, 1536), (1536, 1)),
        cu_seqlens_m=_cu_seqlens(64, 4096),
        cu_seqlens_k=_cu_seqlens(64, 4096),
        device_capability=9,
        dtype="torch.bfloat16",
    )
    assert len(both) == len(pool)


def test_unproven_layouts_fall_back_without_reading_tensor_values():
    pool = _pool("M-group")
    # M-group's packed A is 2-D; a batched A is not covered by the study.
    batched_a = select_varlen_gemm_candidates(
        pool,
        A=_Tensor((2, 4096, 2048), (8192, 2048, 1)),
        B=_Tensor((64, 2048, 1536), (3072, 1536, 1)),
        cu_seqlens_m=_cu_seqlens(64, 4096),
        device_capability=9,
        dtype="torch.bfloat16",
    )
    assert len(batched_a) == len(pool)

    # A 3-D B is supported only when its leading expert axis agrees with the
    # metadata length; an inconsistent layout must not be guessed.
    wrong_expert_axis = select_varlen_gemm_candidates(
        pool,
        A=_Tensor((4096, 2048), (2048, 1)),
        B=_Tensor((32, 2048, 1536), (3072, 1536, 1)),
        cu_seqlens_m=_cu_seqlens(64, 4096),
        device_capability=9,
        dtype="torch.bfloat16",
    )
    assert len(wrong_expert_axis) == len(pool)


def test_selector_leaf_id_uses_metadata_only():
    class MetadataOnly:
        shape = (64,)

        def __len__(self):
            return 64

        def __getattr__(self, name):
            if name in {"detach", "cpu", "cuda", "to", "item", "tolist"}:
                raise AssertionError(f"selector read tensor value via {name}")
            raise AttributeError(name)

    leaf = selector_leaf_id(
        _Tensor((2048, 8192), (1, 2048)),
        _Tensor((8192, 1536), (1536, 1)),
        cu_seqlens_k=MetadataOnly(),
        dtype="torch.bfloat16",
        device_capability=(9, 0),
    )
    assert leaf == "k-low"
    assert (
        selector_leaf_id(
            _Tensor((2048, 8192), (1, 2048)),
            _Tensor((8192, 1536), (1536, 1)),
            cu_seqlens_k=MetadataOnly(),
            dtype="torch.bfloat16",
            device_capability=(10, 0),
        )
        == "fallback"
    )


def test_unsupported_device_dtype_and_out_of_envelope_fall_back():
    workload = _k_workload()
    # 44 static + 2 dynamic twins = the full varlen_k pool
    assert len(_select("K-group", **{**workload, "device_capability": 10})) == 46
    assert len(_select("K-group", **{**workload, "dtype": "torch.float16"})) == 46
    # N outside the measured envelope
    wide = _k_workload(routed_rows=8192, N=16384)
    assert len(_select("K-group", **wide)) == 46
    # expert count outside the measured envelope
    tiny = _k_workload(routed_rows=8192, experts=4)
    assert len(_select("K-group", **tiny)) == 46


@pytest.mark.parametrize("metadata", [{"device_capability": None}, {"dtype": None}])
def test_unknown_selector_metadata_falls_back_to_full_pool(metadata):
    """The rule tree must not prune when validated metadata is unavailable."""
    workload = _k_workload()
    assert len(_select("K-group", **{**workload, **metadata})) == 46


def test_omitted_selector_metadata_falls_back_to_full_pool():
    workload = _k_workload()
    assert len(select_varlen_gemm_candidates(_pool("K-group"), **workload)) == 46


def test_unparseable_device_capability_falls_back_to_full_pool():
    workload = _k_workload()
    assert len(_select("K-group", **{**workload, "device_capability": "unknown"})) == 46


def test_selector_leaf_id_falls_back_for_unknown_metadata():
    workload = _k_workload()
    assert selector_leaf_id(**workload) == FALLBACK_LEAF_ID
    assert (
        selector_leaf_id(**workload, dtype=None, device_capability=9)
        == FALLBACK_LEAF_ID
    )
    assert (
        selector_leaf_id(**workload, dtype="torch.bfloat16", device_capability=None)
        == FALLBACK_LEAF_ID
    )


def test_selector_leaf_is_the_public_decision_probe():
    assert (
        selector_leaf_id(**_k_workload(), dtype="torch.bfloat16", device_capability=9)
        == K_GROUP_LOW_RULE["id"]
    )


def test_workload_metadata_reports_op_and_group_from_real_tensors():
    m_group = workload_metadata(
        _Tensor((4096, 2048), (1, 4096)),
        _Tensor((2048, 1536), (1536, 1)),
        _cu_seqlens(64, 4096),
        None,
    )
    assert m_group["group_type"] == "M-group" and m_group["op"] == "nn"
    nt = workload_metadata(
        _Tensor((2048, 4096), (1, 2048)),
        _Tensor((4096, 1536), (1, 4096)),
        _cu_seqlens(64, 2048),
        None,
    )
    assert nt["group_type"] == "M-group" and nt["op"] == "nt"
    k_group = workload_metadata(
        _Tensor((2048, 8192), (1, 2048)),
        _Tensor((8192, 1536), (1536, 1)),
        None,
        _cu_seqlens(64, 8192),
    )
    assert k_group["group_type"] == "K-group" and k_group["op"] == "tn"
    assert k_group["K"] == 8192 and k_group["N"] == 1536
    assert k_group["expert_count"] == 64
