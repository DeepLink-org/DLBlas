"""Plain-rule candidate pruning for varlen grouped GEMM (SM90, bf16).

What this does
--------------
The varlen grouped-GEMM autotune grid has up to 44 structurally legal
candidates on SM90.  Measuring all of them is what makes the cold start of a
routed MoE step expensive.  This module returns a *subset* of the structurally
legal candidates (bounded by ``KEEP``), picked by a two-leaf decision tree per
family:

* K-group (``op == "tn"``): split on ``K / N``, the total contraction rows per
  unit of the weight's output width;
* M-group (``op in {"nn", "nt"}``): split on ``avg_m``, the routed rows per
  expert.

Each leaf keeps the candidates matching its ``(shape set, cluster set,
swap_ab)`` rule, padded up to ``KEEP`` in :func:`canonical_filler_key` order.
The selector never picks a winner; the autotuner still measures every returned
candidate and keeps the fastest.

Dynamic-persistent twins
-----------------------
Since v7, a leaf may also carry a ``dynamic_shapes`` tile set.  Candidates
with ``is_dynamic_persistent=True`` (appended to the varlen_k pool by
``gemm_interface.prune_invalid_gemm_configs``) enter only a leaf whose
``dynamic_shapes`` explicitly admits their tile; every other leaf keeps its
studied static-only body.  Dynamic variants sort after every static candidate
in :func:`canonical_filler_key`, so leaves without the extension reproduce
their frozen candidate sets byte for byte even when the pool carries twins.

Metadata only
-------------
Every decision is derived from tensor *metadata* - shape, stride, dtype and
the ``cu_seqlens`` *length*.  Nothing here reads a tensor's values, so the
selector never triggers a device-to-host transfer or a sync.  Ragged sizes are
taken from the operand shapes (``A`` is exactly packed to the routed total,
``B`` carries the contraction total in its leading axis); that is the same
layout contract the tuning key already relies on.  Monotonicity of the prefix
sums is therefore not checked here - a malformed ``cu_seqlens`` breaks the
GEMM itself, not the pruning.

Outside the measured envelope (device, dtype, op, N, ragged K, expert count,
or any layout variant the baseline did not cover) the full legal set is
returned unchanged.  The constants are frozen from a measured study; the
acceptance evidence and version history live in
``.vscode/docs/varlen_selector_implementation.md`` (research documentation,
not shipped).
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from torch import Tensor

from cute_gemm.autotuner import AutotuneConfig, _canonical_bench_budget
from cute_gemm.cute_dsl_utils import get_device_capacity
from cute_gemm.gemm_config import get_sm90_dynamic_varlen_configs
from cute_gemm.gemm_tune_policy import GemmTunePolicy

SELECTOR_VERSION = "varlen-rule-tree-v7"

#: Candidate budget.  The autotuner measures at most this many candidates.
KEEP = 8
MIN_RULE_KEEP = 4  # a rule body outside [MIN_RULE_KEEP, KEEP] is unusable

#: Only the measured configuration (SM90, bf16) is covered.  Anything else
#: returns the full structurally legal set.
SUPPORTED_DEVICE_CAPABILITY = 9
SUPPORTED_DTYPE = "bfloat16"

#: Leaf id reported when the selector would not prune a workload.
FALLBACK_LEAF_ID = "fallback"

#: Split feature and the deployed cut of the K-group tree.  The cut sits at the
#: midpoint of a gap in the measured workloads (78.1445 | 80.0), so no workload
#: is near a boundary and small drift cannot move one across it.
K_GROUP_SPLIT_FEATURE = "K_over_N"

#: K-group leaves, evaluated top to bottom against ``K / N``.  Each leaf owns one
#: ``(shape set, cluster set, swap_ab)`` rule; the rule body is padded up to
#: ``KEEP`` by ``canonical_filler_key`` when it matches fewer candidates, which is
#: what the offline study scored.
K_GROUP_TREE: tuple[dict[str, Any], ...] = (
    {
        "id": "k-low",
        "upper": 79.072265625,  # midpoint of the 78.1445 -> 80.0 gap
        "shapes": ((128, 256), (256, 192)),
        "clusters": ((1, 2), (2, 1)),  # both clusters
        "swap_ab": True,
    },
    {
        "id": "k-high",
        "upper": float("inf"),
        "shapes": ((192, 128), (256, 128), (256, 192)),
        "clusters": ((1, 2),),  # cluster (1,2) only, NOT both
        "swap_ab": None,  # both
        # v7: the measured SM90 dynamic-persistent winners (long-tail routed K,
        # -13.5%/-14.5% on the production wgrad shape, bitwise-identical).
        # Appended to the 6 static matches they fill the KEEP budget exactly.
        "dynamic_shapes": ((256, 128),),
    },
)

#: M-group split feature: mean ragged rows per expert (routed rows / expert
#: count, empty experts included).
M_GROUP_SPLIT_FEATURE = "avg_m"

#: M-group leaves (op in {nn, nt}), evaluated top to bottom against ``avg_m``.
#: Both bodies keep 7 candidates and are topped up to 8.  The cut sits in the
#: widest gap of the axis (1024.0 | 1365.33).
M_GROUP_TREE: tuple[dict[str, Any], ...] = (
    {
        "id": "m-low",
        "upper": 1194.6666666666667,  # midpoint of the 1024.0 -> 1365.3333 gap
        "shapes": (
            (128, 160),
            (128, 192),
            (128, 208),
            (128, 224),
            (128, 256),
            (192, 128),
            (256, 128),
        ),
        "clusters": ((1, 2),),
        "swap_ab": False,
    },
    {
        "id": "m-high",
        "upper": float("inf"),
        "shapes": (
            (128, 192),
            (128, 256),
            (192, 128),
            (256, 128),
            (256, 160),
            (256, 192),
            (256, 208),
        ),
        "clusters": ((1, 2),),
        "swap_ab": False,
    },
)

#: Measured envelope (min/max over the 322 baseline cases).  Outside it the
#: selector returns the full structurally legal set instead of guessing.
TRAINED_ENVELOPE: dict[str, dict[str, Any]] = {
    "K-group": {
        "ops": ("tn",),
        "N": (512, 7168),
        "ragged_K": (8, 4098424),
        "expert_count": (32, 768),
    },
    "M-group": {
        "ops": ("nt", "nn"),
        "N": (512, 7168),
        "ragged_K": (512, 7168),
        "expert_count": (32, 768),
    },
}


#: Accepted spellings of the one dtype this rule table was measured on.
DTYPE_ALIASES = {"bfloat16": "bfloat16", "bf16": "bfloat16"}


def _normalise_dtype(dtype: Any) -> str:
    """Torch dtype or any common spelling of it, as one comparable string."""
    name = (getattr(dtype, "name", None) or str(dtype)).strip().lower()
    name = name.removeprefix("torch.")
    return DTYPE_ALIASES.get(name, name)


def _capability_major(device_capability: Any) -> int | None:
    """Accept either a major integer or a cached ``(major, minor)`` tuple."""
    if device_capability is None:
        return None
    try:
        major = (
            device_capability[0]
            if isinstance(device_capability, (tuple, list))
            else device_capability
        )
        return int(major)
    except (IndexError, TypeError, ValueError):
        # Treat malformed capability metadata as unknown and let callers use
        # the conservative full-candidate fallback.
        return None


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def _coverage_reason(workload: dict[str, Any]) -> str | None:
    """None when the workload is inside the measured envelope."""
    envelope = TRAINED_ENVELOPE.get(workload["group_type"])
    if envelope is None:
        return f"group_type {workload['group_type']!r} not in training coverage"
    if workload["op"] not in envelope["ops"]:
        return f"op {workload['op']!r} not in training coverage for {workload['group_type']}"
    if not _in_range(workload["N"], envelope["N"]):
        return f"N={workload['N']} outside trained range {envelope['N']}"
    if not _in_range(workload["K"], envelope["ragged_K"]):
        return f"ragged K={workload['K']} outside trained range {envelope['ragged_K']}"
    if not _in_range(workload["expert_count"], envelope["expert_count"]):
        return (
            f"expert_count={workload['expert_count']} outside trained range "
            f"{envelope['expert_count']}"
        )
    return None


def _length_of(values: Any) -> int:
    """Number of entries a cu_seqlens prefix-sum carries, from metadata only.

    Tensors expose ``shape``; plain sequences fall back to ``len`` so test
    doubles work.  Neither reads a value.
    """
    shape = getattr(values, "shape", None)
    return len(values) if shape is None else int(shape[0])


def _shape_of(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    return tuple(int(size) for size in shape) if shape is not None else ()


def _stride_of(tensor: Any) -> tuple[int, ...]:
    stride = getattr(tensor, "stride", None)
    if not callable(stride):
        return ()
    try:
        return tuple(int(value) for value in stride())
    except (TypeError, RuntimeError):
        return ()


def workload_metadata(A, B, cu_seqlens_m, cu_seqlens_k, A_idx=None) -> dict[str, Any]:
    """Derive op/group/N/K/expert-count from operand metadata alone.

    The ragged sizes come from the operand shapes and the segment count from
    the ``cu_seqlens`` length; the values of ``cu_seqlens`` are never read.
    """
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    if varlen_m and varlen_k:
        return {
            "valid": False,
            "varlen": True,
            "reason": "both cu_seqlens_m and cu_seqlens_k were provided",
        }
    if not (varlen_m or varlen_k):
        return {"valid": False, "varlen": False, "reason": "not a varlen grouped GEMM"}
    if A is None or B is None:
        return {"valid": False, "varlen": True, "reason": "missing A or B operand"}
    if A_idx is not None:
        return {
            "valid": False,
            "varlen": True,
            "reason": "gather_A layout is not covered by the baseline study",
        }
    cu_seqlens = cu_seqlens_m if varlen_m else cu_seqlens_k
    expert_count = _length_of(cu_seqlens) - 1
    if expert_count < 1:
        return {
            "valid": False,
            "varlen": True,
            "reason": "cu_seqlens must carry at least one expert segment",
        }
    a_shape, b_shape = _shape_of(A), _shape_of(B)
    if len(a_shape) < 2 or len(b_shape) < 2:
        return {"valid": False, "varlen": True, "reason": "operands are not 2-D/value tensors"}
    if varlen_m:
        # The measured M-group layout packs routed rows in A.  B is either a
        # shared (K, N) weight or an expert-major (E, K, N) weight; for the
        # latter, its leading extent must agree with cu_seqlens metadata.
        if len(a_shape) != 2 or len(b_shape) not in (2, 3):
            return {
                "valid": False,
                "varlen": True,
                "reason": "M-group layout is outside the packed 2-D/3-D contract",
            }
        if len(b_shape) == 3 and b_shape[0] != expert_count:
            return {
                "valid": False,
                "varlen": True,
                "reason": "expert-major B does not match cu_seqlens length",
            }
        group = "M-group"
        N, K, routed_rows = b_shape[-1], a_shape[-1], a_shape[-2]
        op = "nn" if (_stride_of(B) and _stride_of(B)[-1] == 1) else "nt"
    else:
        # K-group uses the packed 2-D B=(total_K, N) contraction operand.
        if len(a_shape) != 2 or len(b_shape) != 2:
            return {
                "valid": False,
                "varlen": True,
                "reason": "K-group layout is outside the packed 2-D contract",
            }
        group = "K-group"
        op = "tn"
        # B is (total_K, N): the contraction total is its leading axis.
        N, K, routed_rows = b_shape[-1], b_shape[0], b_shape[0]
    return {
        "valid": True,
        "group_type": group,
        "op": op,
        "N": int(N),
        "K": int(K),
        "routed_rows": int(routed_rows),
        "expert_count": int(expert_count),
        "A_shape": a_shape,
        "B_shape": b_shape,
        "A_stride": _stride_of(A),
        "B_stride": _stride_of(B),
    }


def _leaf_decision(workload: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
    """(rule, split_feature, split_value) for a valid, in-envelope workload."""
    if workload["group_type"] == "K-group":
        N = int(workload["N"])
        value = (int(workload["K"]) / N) if N > 0 else float("inf")
        tree, feature = K_GROUP_TREE, K_GROUP_SPLIT_FEATURE
    else:
        value = int(workload["routed_rows"]) / int(workload["expert_count"])
        tree, feature = M_GROUP_TREE, M_GROUP_SPLIT_FEATURE
    for leaf in tree:
        if value <= leaf["upper"]:
            return leaf, feature, value
    return tree[-1], feature, value


def selector_leaf_id(
    A,
    B,
    cu_seqlens_m=None,
    cu_seqlens_k=None,
    A_idx=None,
    dtype=None,
    device_capability=None,
    selector_enabled=True,
) -> str:
    """Which leaf's candidate set a workload would be tuned on - metadata only.

    The tuning key embeds this id, so two workloads that share a cache bucket
    but fall on different tree leaves tune separately instead of the first
    arrival deciding for the whole bucket, and anything the selector would
    return unpruned (out of envelope, gather-A, ...) shares ``FALLBACK_LEAF_ID``
    rather than a leaf.  Uses the same rule tables and envelope as
    :func:`select_varlen_gemm_candidates`, so the key and the pruning decision
    cannot drift apart.  ``device_capability`` accepts either a major integer
    or the cached ``(major, minor)`` tuple.
    """
    if not selector_enabled:
        return FALLBACK_LEAF_ID
    workload = workload_metadata(A, B, cu_seqlens_m, cu_seqlens_k, A_idx)
    if not workload["valid"]:
        return FALLBACK_LEAF_ID
    if dtype is None:
        return FALLBACK_LEAF_ID
    if _normalise_dtype(dtype) != SUPPORTED_DTYPE:
        return FALLBACK_LEAF_ID
    major = _capability_major(device_capability)
    if device_capability is None or major is None:
        return FALLBACK_LEAF_ID
    if major != SUPPORTED_DEVICE_CAPABILITY:
        return FALLBACK_LEAF_ID
    if _coverage_reason(workload) is not None:
        return FALLBACK_LEAF_ID
    return _leaf_decision(workload)[0]["id"]


def candidate_fields(config: Any) -> dict[str, Any]:
    """Read tile/cluster fields from an AutotuneConfig or a bare GemmConfig."""
    kwargs = getattr(config, "kwargs", None)
    if isinstance(kwargs, dict) and "config" in kwargs:
        config = kwargs["config"]
    return {
        "tile_m": int(config.tile_m),
        "tile_n": int(config.tile_n),
        "tile_k": None if config.tile_k is None else int(config.tile_k),
        "num_warps": (None if config.num_warps is None else int(config.num_warps)),
        "cluster_m": int(config.cluster_m),
        "cluster_n": int(config.cluster_n),
        "pingpong": bool(config.pingpong),
        "swap_ab": bool(config.swap_ab),
        "split_k": int(getattr(config, "split_k", 1)),
        "is_dynamic_persistent": bool(getattr(config, "is_dynamic_persistent", False)),
    }


def canonical_filler_key(fields: dict[str, Any]) -> tuple[Any, ...]:
    """Fixed, label-free order used to fill a rule body up to ``KEEP``.

    Smallest tile area first - a grouped GEMM whose routed rows barely cover a
    tile wants the small shapes - then a total order on the remaining fields so
    the result is reproducible.  This order was part of what the offline study
    scored, so treat it as a frozen design constant.  The leading
    ``is_dynamic_persistent`` flag was added after that study (v7) and sorts
    every dynamic variant after all static ones, so the studied static-vs-static
    order - and therefore every leaf's filler composition - is unchanged.
    """
    tile_m, tile_n = int(fields["tile_m"]), int(fields["tile_n"])
    tile_k = -1 if fields["tile_k"] is None else int(fields["tile_k"])
    num_warps = -1 if fields["num_warps"] is None else int(fields["num_warps"])
    return (
        int(bool(fields["is_dynamic_persistent"])),
        tile_m * tile_n,
        tile_m,
        tile_n,
        int(fields["cluster_m"]),
        int(fields["cluster_n"]),
        int(bool(fields["swap_ab"])),
        tile_k,
        num_warps,
        int(bool(fields["pingpong"])),
    )


def _rule_keeps(rule: dict[str, Any], fields: dict[str, Any]) -> bool:
    if bool(fields["is_dynamic_persistent"]):
        # A dynamic-persistent candidate only enters a leaf whose
        # ``dynamic_shapes`` explicitly admits its tile; every other leaf keeps
        # its studied static-only body.
        dynamic_shapes = rule.get("dynamic_shapes")
        if dynamic_shapes is None:
            return False
        if (int(fields["tile_m"]), int(fields["tile_n"])) not in dynamic_shapes:
            return False
    if (int(fields["tile_m"]), int(fields["tile_n"])) not in rule["shapes"]:
        return False
    if (int(fields["cluster_m"]), int(fields["cluster_n"])) not in rule["clusters"]:
        return False
    return rule["swap_ab"] is None or bool(fields["swap_ab"]) is bool(rule["swap_ab"])


def _log_decision(record: dict[str, Any]) -> None:
    path = os.environ.get("DLKERNEL_VARLEN_SELECTOR_LOG")
    if os.environ.get("DLKERNEL_DEBUG_TUNE") == "1":
        print(
            "DLKERNEL_VARLEN_SELECTOR "
            f"version={record.get('selector_version')} applied={record.get('applied')} "
            f"fallback={record.get('fallback')} original={record.get('original_candidate_count')} "
            f"kept={record.get('selected_candidate_count')} rule={record.get('rule')} "
            f"reason={record.get('reason')}",
            flush=True,
        )
    if not path:
        return
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def select_varlen_gemm_candidates(
    configs: Sequence[Any],
    *,
    A=None,
    B=None,
    cu_seqlens_m=None,
    cu_seqlens_k=None,
    A_idx=None,
    device_capability: int | tuple[int, int] | None = None,
    dtype=None,
) -> list[Any]:
    """Return the rule-tree candidate subset for varlen grouped GEMM.

    Dense GEMM, unsupported devices/dtypes and any structural variant the
    baseline study did not cover return ``configs`` unchanged; the autotuner
    still measures every returned candidate.  The result is always a sub-list
    of ``configs``, never empty, and never reordered relative to the canonical
    fill order (so it does not depend on the caller's ordering).
    """
    record: dict[str, Any] = {
        "selector_version": SELECTOR_VERSION,
        "applied": False,
        "fallback": False,
        "original_candidate_count": len(configs),
        "selected_candidate_count": len(configs),
        "reason": None,
    }

    def finish(result: list[Any]) -> list[Any]:
        record["selected_candidate_count"] = len(result)
        _log_decision(record)
        return result

    workload = workload_metadata(A, B, cu_seqlens_m, cu_seqlens_k, A_idx)
    if not workload["valid"]:
        record["reason"] = workload["reason"]
        record["fallback"] = bool(workload.get("varlen"))
        return finish(list(configs))

    record["features"] = dict(workload)
    record["op"] = workload["op"]
    record["group_type"] = workload["group_type"]

    if dtype is None:
        record.update({"fallback": True, "reason": "dtype metadata unavailable"})
        return finish(list(configs))
    if _normalise_dtype(dtype) != SUPPORTED_DTYPE:
        record.update({"fallback": True, "reason": f"dtype {dtype} not in validated coverage"})
        return finish(list(configs))
    major = _capability_major(device_capability)
    if device_capability is None or major is None:
        record.update({"fallback": True, "reason": "device capability metadata unavailable"})
        return finish(list(configs))
    if major != SUPPORTED_DEVICE_CAPABILITY:
        record.update(
            {
                "fallback": True,
                "reason": f"device_capability {device_capability} not in validated coverage",
            }
        )
        return finish(list(configs))

    uncovered = _coverage_reason(workload)
    if uncovered is not None:
        record.update({"fallback": True, "reason": uncovered})
        return finish(list(configs))

    if len(configs) <= KEEP:
        record["reason"] = "candidate set already within the budget"
        return finish(list(configs))

    rule, feature, value = _leaf_decision(workload)
    record["rule"] = rule["id"]
    record["split_feature"] = feature
    record["split_value"] = value
    record["split_threshold"] = rule["upper"]
    record["applied"] = True

    ordered = sorted(configs, key=lambda config: canonical_filler_key(candidate_fields(config)))
    body = [config for config in ordered if _rule_keeps(rule, candidate_fields(config))]
    record["rule_matched_count"] = len(body)
    if not MIN_RULE_KEEP <= len(body) <= KEEP:
        # The rule no longer fits this kernel grid (too small a body, or a
        # grid where it matches more than the budget).  Falling back to the
        # full set is the safe move: measure everything, prune nothing.
        record.update(
            {
                "applied": False,
                "fallback": True,
                "reason": (
                    f"rule matched {len(body)} candidates on this kernel grid, "
                    f"outside the [{MIN_RULE_KEEP}, {KEEP}] budget"
                ),
            }
        )
        return finish(list(configs))
    if len(body) == KEEP:
        return finish(body)
    fillers = [config for config in ordered if config not in body]
    return finish(body + fillers[: KEEP - len(body)])


def _rule_public_view(rule: dict[str, Any]) -> dict[str, Any]:
    view = {
        "id": rule["id"],
        "upper": rule.get("upper"),
        "shapes": [f"{m}x{n}" for m, n in rule["shapes"]],
        "clusters": [f"{m},{n}" for m, n in rule["clusters"]],
        "swap_ab": rule["swap_ab"],
    }
    if "dynamic_shapes" in rule:
        view["dynamic_shapes"] = [f"{m}x{n}" for m, n in rule["dynamic_shapes"]]
    return view


def rule_table() -> dict[str, Any]:
    """The frozen constants, for tests and for the acceptance report."""
    return {
        "selector_version": SELECTOR_VERSION,
        "keep": KEEP,
        "device_capability": SUPPORTED_DEVICE_CAPABILITY,
        "dtype": SUPPORTED_DTYPE,
        "split_feature": K_GROUP_SPLIT_FEATURE,
        "m_split_feature": M_GROUP_SPLIT_FEATURE,
        "k_tree": [_rule_public_view(leaf) for leaf in K_GROUP_TREE],
        "m_tree": [_rule_public_view(leaf) for leaf in M_GROUP_TREE],
        "envelope": TRAINED_ENVELOPE,
    }


# =========================================================================== #
# Varlen bucket winner key (metadata-only; never reads device tensor values)
# =========================================================================== #

#: Smallest average-rows bucket.  MoE shards in the measured envelope start at
#: 32 routed rows per expert; a finer floor would only split the key space.
_AVERAGE_ROWS_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
# The last label is the open-ended >1024 bucket, not an upper bound of 2048.
_MIN_AVERAGE_ROWS = _AVERAGE_ROWS_BUCKETS[0]

#: Prefix of the key.  v3 includes the selector leaf, canonical defaults and the
#: operand device identity.  v4 adds the dynamic-persistent k-high candidates;
#: v5 additionally canonicalizes routed epilogue vectors such as
#: ``mColVecBroadcast`` with the routed-M bucket. v6 caps average rows at
#: the open >1024 tail. Each bump deliberately
#: orphans older winners whose identity did not carry the new distinction.
_KEY_VERSION = "gemm_token_bucket_v6"
_LEGACY_KEY_VERSION = "gemm_token_bucket_v5"

#: Named operands.  They are covered by the shape rewrite below, so they are
#: never added to the key as plain values (their identity does not matter).
_OPERAND_NAMES = ("A", "B", "out", "C")

#: Marks a parameter that was not passed at all (defaulted).  A value of
#: ``None`` *was* passed and still belongs in the key, like v1's merged dict.
_MISSING = object()

#: Parameter names that switch miss behaviour, not workload identity.  They
#: must never enter W or F; both collection loops below exclude them.
_NON_IDENTITY_NAMES = frozenset({"tune_mode"})

#: Prefix of the fixed-contract identity F.  Distinct from ``_KEY_VERSION`` so
#: an F tuple can never be confused with (or alias) a W tuple.
_CONTRACT_VERSION = "gemm_contract_v1"

#: Constant extent a declared ragged dimension collapses to inside F.
_RAGGED_MARK = -1


@lru_cache(maxsize=8)
def _argument_index(arg_names: tuple[str, ...]) -> dict[str, int]:
    """Position of each named parameter, cached per decorated signature."""
    return {name: index for index, name in enumerate(arg_names)}


@lru_cache(maxsize=8)
def _key_layout(arg_names: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    """(name, position) pairs in the key's canonical order.

    Alphabetical by name, which is the order v1's per-call ``sorted()`` produced;
    caching it keeps the key layout byte-identical without sorting on the
    cache-hit path.
    """
    return tuple(sorted((name, index) for index, name in enumerate(arg_names)))


def average_rows_bucket(rows: int, experts: int, *, device_capability=None) -> int:
    """Power-of-two average rows; SM90 has six bounded bins and one >1024 bin.

    ``-1`` marks an unusable expert count so the caller can still key on it.
    """
    if experts <= 0:
        return -1
    average = (rows + experts - 1) // experts
    if average <= _MIN_AVERAGE_ROWS:
        return _MIN_AVERAGE_ROWS
    # Smallest power of two >= average, without a doubling loop.
    bucket = 1 << (average - 1).bit_length()
    return (
        min(bucket, _AVERAGE_ROWS_BUCKETS[-1])
        if device_capability and device_capability[0] == 9
        else bucket
    )


def _ragged_dimension(rows: int, experts: int, device_capability=None) -> int:
    """Collapsed size of a ragged dimension, still monotone in its real size."""
    bucket = average_rows_bucket(rows, experts, device_capability=device_capability)
    return -1 if bucket < 0 else bucket * max(experts, 1)


def _reuse_profile(stride: tuple[int, ...]) -> tuple[int, ...]:
    """Standardised stride: 0/1 are kept, anything else is flattened to 2."""
    return tuple(min(2, size) for size in stride)


def _device_capability(tensor: Any):
    """Return the cached capability without materialising or moving a tensor.

    CPU-only tests may provide ``DLKERNEL_ARCH``; a real CUDA tensor uses the
    repository's cached capability helper.  If neither is available, the
    conservative fallback leaf is used rather than guessing a rule leaf.
    """
    device = getattr(tensor, "device", None)
    if device is None:
        return None
    if getattr(device, "type", None) != "cuda" and os.environ.get("DLKERNEL_ARCH") is None:
        return None
    try:
        return get_device_capacity(device)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None


def _arg_value(args, kwargs, name: str, position: int | None, default: Any = _MISSING) -> Any:
    """The value bound to ``name``: keyword beats position, then default."""
    if name in kwargs:
        return kwargs[name]
    if position is not None and position < len(args):
        return args[position]
    return default


def gemm_tune_key(
    args,
    kwargs,
    arg_names,
    arg_defaults=None,
    ragged_m_out_names=("out", "C", "D"),
    opaque_tensor_names=(),
    routed_m_tensor_names=(),
    device_capability=None,
) -> tuple[Any, ...] | None:
    """Bucket-collapsed varlen tune key (W); ``None`` for dense calls.

    ``ragged_m_out_names``: output-style tensors whose ``shape[-2]`` tracks the
    ragged M and is rewritten to the bucketed size ("D" is the epilogue tuner's
    output alias; plain GEMM has no such parameter, so the default is inert).

    ``opaque_tensor_names``: caller-owned scratch whose exact shape derives
    from the ragged size and is re-derived per call (reduce-sink buffers,
    bundled transform operands).  Only ``dtype`` enters the key; capacity
    legality is enforced per call by the operator's prune, not by winner
    identity.

    ``routed_m_tensor_names``: per-routed-row vectors (for example
    ``mColVecBroadcast`` in a fused MoE epilogue).  Their row extent is
    canonicalized with the same routed-M bucket as ``A``/the ragged outputs;
    runtime shape and buffer legality remain checked by the operator.
    """
    return _build_identity(
        args,
        kwargs,
        arg_names,
        arg_defaults,
        ragged_m_out_names,
        opaque_tensor_names,
        routed_m_tensor_names,
        contract=False,
        device_capability=device_capability,
    )


def gemm_contract_key(
    args,
    kwargs,
    arg_names,
    arg_defaults=None,
    ragged_m_out_names=("out", "C", "D"),
    opaque_tensor_names=(),
    routed_m_tensor_names=(),
    device_capability=None,
) -> tuple[Any, ...] | None:
    """Fixed-contract identity F; ``None`` for dense calls.

    Same single metadata pass and tensor-role declarations as
    :func:`gemm_tune_key`, but the declared token dimensions collapse to a
    constant mark instead of their bucket, and the measurement-side components
    (avg-expert buckets, selector leaf, bench budget) are absent: F names the
    operator *contract* — weights geometry, expert count, dtype, layout,
    epilogue identity — not the measured workload point.  Calls that differ
    only in routed totals share F; changing E/N/K/dtype/layout forms a new F.
    Like W, F never contains ``tune_mode``, tensor addresses, or tensor values.
    """
    return _build_identity(
        args,
        kwargs,
        arg_names,
        arg_defaults,
        ragged_m_out_names,
        opaque_tensor_names,
        routed_m_tensor_names,
        contract=True,
        device_capability=device_capability,
    )


def _build_identity(
    args,
    kwargs,
    arg_names,
    arg_defaults,
    ragged_m_out_names,
    opaque_tensor_names,
    routed_m_tensor_names,
    *,
    contract: bool,
    device_capability=None,
):
    names = tuple(arg_names)
    ragged_m_out_names = frozenset(ragged_m_out_names)
    opaque_names = frozenset(opaque_tensor_names)
    routed_m_tensor_names = frozenset(routed_m_tensor_names)
    positions = _argument_index(names)
    if arg_defaults is None:
        defaults = {}
    else:
        defaults = {
            name: default
            for name, default in zip(names, arg_defaults)
            if default is not _MISSING and default is not inspect.Parameter.empty
        }

    def bound_value(name: str) -> Any:
        default = defaults.get(name, _MISSING)
        return _arg_value(args, kwargs, name, positions.get(name), default)

    # A parameter missing from the signature binds to the _MISSING sentinel
    # (not None): the epilogue tuner's signature has cu_seqlens_m but not
    # cu_seqlens_k, and that absence must not read as "varlen K call".
    cu_m = bound_value("cu_seqlens_m")
    cu_k = bound_value("cu_seqlens_k")
    varlen_m = cu_m is not None and cu_m is not _MISSING
    varlen_k = cu_k is not None and cu_k is not _MISSING
    if not (varlen_m or varlen_k):
        return None

    def bound(name: str) -> Any:
        value = bound_value(name)
        return None if value is _MISSING else value

    # One pass over the arguments in canonical order collects scalars and
    # tensors alike.  v1 merged a dict and ran two sorted() passes per call;
    # this runs on every cache-hit, so it stays allocation-light.
    a = bound("A")
    capability = (
        device_capability
        if device_capability is not None
        else _device_capability(a)
        if isinstance(a, Tensor)
        else None
    )
    key_version = _KEY_VERSION if capability and capability[0] == 9 else _LEGACY_KEY_VERSION
    key: list = [_CONTRACT_VERSION if contract else key_version]
    tensors: list = []
    for name, position in _key_layout(names):
        value = _arg_value(args, kwargs, name, position, defaults.get(name, _MISSING))
        if value is _MISSING:
            continue
        if isinstance(value, Tensor):
            tensors.append((name, value))
        elif name not in _OPERAND_NAMES and name not in _NON_IDENTITY_NAMES:
            key.append((name, value))
    for name in sorted(kwargs.keys() - positions.keys()):
        value = kwargs[name]
        if isinstance(value, Tensor):
            tensors.append((name, value))
        elif name not in _OPERAND_NAMES and name not in _NON_IDENTITY_NAMES:
            key.append((name, value))

    cu_seqlens_m = bound("cu_seqlens_m")
    cu_seqlens_k = bound("cu_seqlens_k")
    a_index = bound("A_idx")
    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    gather_a = a_index is not None
    a, b = bound("A"), bound("B")
    m_experts = cu_seqlens_m.shape[0] - 1 if varlen_m else 0
    k_experts = cu_seqlens_k.shape[0] - 1 if varlen_k else 0
    # With gather_A the index length is the routed total; without it the ragged
    # size is readable straight off the operand.
    routed_m = (
        a_index.shape[0]
        if varlen_m and gather_a
        else a.shape[-2]
        if varlen_m and isinstance(a, Tensor)
        else 0
    )
    routed_k = (
        a_index.shape[0]
        if varlen_k and gather_a
        else b.shape[0]
        if varlen_k and isinstance(b, Tensor)
        else 0
    )

    def _collapse(size: int, experts: int) -> int:
        """Ragged extent: its monotone bucket for W, a constant mark for F."""
        return _RAGGED_MARK if contract else _ragged_dimension(size, experts, capability)

    if not contract:
        if varlen_m and isinstance(a, Tensor):
            key.append(
                (
                    "avg_expert_m",
                    average_rows_bucket(routed_m, m_experts, device_capability=capability),
                )
            )
        if varlen_k and isinstance(b, Tensor):
            key.append(
                (
                    "avg_expert_k",
                    average_rows_bucket(routed_k, k_experts, device_capability=capability),
                )
            )

    if isinstance(a, Tensor):
        device = getattr(a, "device", None)
        if device is not None:
            if capability is not None:
                key.append(("device_capability", tuple(capability)))
            else:
                # Device ordinal is intentionally excluded.  CUDA_VISIBLE_DEVICES
                # remapping must not split winners for equivalent hardware.
                key.append(("device_type", device.type))

    for name, value in tensors:
        if name in opaque_names:
            # Caller-owned scratch derived from the ragged size (reduce sinks,
            # transform bundles): dtype only. Sink capacity is enforced at the
            # tuned-call boundary, not by cache identity or candidate pruning.
            key.append((name, value.dtype))
            continue
        shape = list(value.shape)
        # Only declared roles may collapse M; equal extents do not imply equal roles.
        if varlen_m and name in ragged_m_out_names and len(shape) >= 2:
            shape[-2] = _collapse(shape[-2], m_experts)
        elif varlen_m and name in routed_m_tensor_names and len(shape) >= 1:
            # mColVecBroadcast is a per-routed-row vector.  Use its final
            # extent for both (M,) and (L, M) spellings, while leaving the
            # actual tensor shape available to runtime legality checks.
            shape[-1] = _collapse(shape[-1], m_experts)
        elif varlen_m and name == "A" and not gather_a and len(shape) >= 2:
            shape[-2] = _collapse(shape[-2], m_experts)
        if varlen_m and name == "A_idx" and len(shape) >= 1:
            shape[-1] = _collapse(shape[-1], m_experts)
        # Ragged-K layout: B's leading axis, and A's last axis unless gather_A
        # carries total K in the index instead.
        if varlen_k and name == "A" and not gather_a and len(shape) >= 2:
            shape[-1] = _collapse(shape[-1], k_experts)
        if varlen_k and name == "B" and len(shape) >= 2:
            shape[0] = _collapse(shape[0], k_experts)
        if varlen_k and name == "A_idx" and len(shape) >= 1:
            shape[-1] = _collapse(shape[-1], k_experts)
        key.extend((name, tuple(shape), _reuse_profile(value.stride()), value.dtype))

    if not contract:
        # The rule-tree leaf the selector would prune this workload to.  Same
        # metadata inputs as the pruning decision, so the key and the selector
        # cannot disagree about which candidates a bucket entry was tuned on.
        key.append(
            (
                "selector_leaf",
                selector_leaf_id(
                    a,
                    b,
                    cu_seqlens_m,
                    cu_seqlens_k,
                    a_index,
                    dtype=getattr(a, "dtype", None),
                    device_capability=capability,
                    selector_enabled=os.environ.get("DLKERNEL_VARLEN_SELECTOR", "1") != "0",
                ),
            )
        )
        # Bench protocol settings affect the measured winner.  Key the canonical
        # effective values so equivalent spellings/clamps cannot create aliases.
        warmup_ms, timed_calls = _canonical_bench_budget()
        key.append(
            (
                "bench_budget_env",
                warmup_ms,
                timed_calls,
            )
        )
    return tuple(key)


class VarlenGemmTunePolicy(GemmTunePolicy):
    """GEMM policy specialization for grouped/variable-length workloads.

    ``GemmTunePolicy`` owns the generic GEMM structural phase.  This subclass
    owns the variable-length facts that are deliberately absent from that
    phase: bucket identity, selector leaf identity, and measured shortlist.
    Dense calls delegate to the parent unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ragged_m_out_names = ("out", "C", "D")
        self._opaque_tensor_names = ()
        self._routed_m_tensor_names = ()

    def _gemm_key(self, args, kwargs, *, contract=False, device_capability=None):
        builder = gemm_contract_key if contract else gemm_tune_key
        return builder(
            args,
            kwargs,
            self.arg_names,
            self.arg_defaults,
            ragged_m_out_names=self._ragged_m_out_names,
            opaque_tensor_names=self._opaque_tensor_names,
            routed_m_tensor_names=self._routed_m_tensor_names,
            device_capability=device_capability,
        )

    def running_configs(self, args, kwargs, configs):
        from cute_gemm.gemm_running import running_configs

        return running_configs(self, args, kwargs, configs)

    def prepares_running(self, key):
        return isinstance(key, tuple) and bool(key) and key[0] == _KEY_VERSION

    def bind_signature(self, arg_names, arg_defaults) -> None:
        super().bind_signature(arg_names, arg_defaults)
        self._varlen_params = bool({"cu_seqlens_m", "cu_seqlens_k"}.intersection(self.arg_names))

    def prepare_candidates(self, configs, named_args, kwargs):
        merged = dict(named_args)
        merged.update(kwargs)
        if merged.get("cu_seqlens_k") is not None:
            configs = [
                *configs,
                *(AutotuneConfig(config=config) for config in get_sm90_dynamic_varlen_configs()),
            ]
        return super().prepare_candidates(configs, named_args, kwargs)

    def make_key(self, args, kwargs, default_key):
        if not getattr(self, "_varlen_params", False):
            return default_key()
        key = self._gemm_key(args, kwargs)
        return default_key() if key is None else key

    def make_contract(self, args, kwargs, default_key):
        """Fixed operator contract F: W minus token buckets/leaf/budget.

        Dense calls keep their full default-key geometry as the contract
        (dense has no declared dynamic dimension to fold).
        """
        if not getattr(self, "_varlen_params", False):
            return default_key()
        contract = self._gemm_key(args, kwargs, contract=True)
        return default_key() if contract is None else contract

    def budget_override(self, key):
        return (
            isinstance(key, tuple) and bool(key) and key[0] in (_KEY_VERSION, _LEGACY_KEY_VERSION)
        )

    def shortlist(self, configs, named_args, kwargs):
        # The parent owns dense structural/performance staging.  Calling it
        # first keeps this operator-specific selector from accidentally
        # bypassing split-K policy semantics as the candidate pipeline grows.
        configs = super().shortlist(configs, named_args, kwargs)
        merged = dict(named_args)
        merged.update(kwargs)
        cu_m = merged.get("cu_seqlens_m")
        cu_k = merged.get("cu_seqlens_k")
        if cu_m is None and cu_k is None:
            return configs
        if os.environ.get("DLKERNEL_VARLEN_SELECTOR", "1") == "0":
            return configs
        A = merged.get("A")
        if A is None:
            return configs
        try:
            device_capacity = get_device_capacity(A.device)[0]
        except (AssertionError, RuntimeError, TypeError, ValueError):
            device_capacity = None
        selected = select_varlen_gemm_candidates(
            configs,
            A=A,
            B=merged.get("B"),
            cu_seqlens_m=cu_m,
            cu_seqlens_k=cu_k,
            A_idx=merged.get("A_idx"),
            device_capability=device_capacity,
            dtype=getattr(A, "dtype", None),
        )
        if not selected:
            raise RuntimeError("varlen candidate selector returned an empty candidate set")
        return selected
