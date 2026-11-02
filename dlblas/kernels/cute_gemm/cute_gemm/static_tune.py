# Copyright (c) 2026, Tri Dao.
"""Offline tuner CLI: ``python -m cute_gemm.static_tune {study,production,prefill}``.

One case enumerator, three modes (plan v4 §4):

* ``study``    — TuneMode.TUNED with the varlen selector OFF, sweeping the
  prepared full pool.  Each raw case gets its own ``DLKERNEL_CACHE_DIR``
  (same-bucket cases must not reuse each other's timings) while the
  artifact (.o) cache stays shared.  Harvested timings go to a dataset
  file; study winners are NOT deployable for varlen (their key carries the
  ``fallback`` selector leaf).
* ``production`` — TuneMode.TUNED with deployment selector/budget settings;
  the winners it writes are the deployable measured entries.
* ``prefill``  — resolve a legal config per case (no measurement), then
  drive ``cute_gemm.cache.precompile.precompile()`` to fill the shared .o
  artifact cache.

The manifest is a JSON list of cases; each case is a dict with ``family``
(``dense`` | ``varlen_m`` | ``varlen_k`` | ``fused_up`` | ``fused_down`` |
``epi``) and the dims it needs (``m``/``n``/``k``, ``experts``, ``tokens`` for
the ragged total, ``dtype``).  Dims stay on those shared keys so the case slug
covers them: the fused families contract over ``k`` and take the literal op
width from ``n`` (swiglu concat rows 2I for up, hidden H for down; keep E and
dims small -- E=8, three-digit extents sweep cheap).  ``epi`` is dense-shaped
plus the fp32 ``[1, n]`` bias its op consumes.
GPU sharding is the launcher's job: run N shard processes with pinned
``CUDA_VISIBLE_DEVICES`` and ``--shard i/N``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cute_gemm.epilogue.frontend import gemm_epilogue

# Operator-owned families: the heuristic ranker has no impl for them, so
# prefill resolves them through the operator's own STATIC chain (see
# _prefill_call) and study/production call the public entry directly.
_OP_OWNED_FAMILIES = ("fused_up", "fused_down", "epi")


@gemm_epilogue(outputs=("doubled",))
def _epi_bias_double(acc, bias):
    """Representative ``epi``-family op: tests/test_epilogue_iface.py's
    bias-double (biased D plus one extra full-buffer output).  Study sweeps it
    through the tuned path; prefill resolves it STATIC — the ranker
    deliberately has no epi impl."""
    y = acc + bias
    return {"D": y, "doubled": y * 2.0}


def _iter_cases(manifest: Path, shard: str | None):
    cases = json.loads(manifest.read_text())
    if shard:
        index, total = (int(x) for x in shard.split("/"))
        cases = cases[index::total]
    return cases


def _run_case(case: dict, mode: str, cache_root: Path, out_root: Path | None):
    """Execute one case under a per-case winner dir; return its record.

    Swapping ``DLKERNEL_CACHE_DIR`` per case isolates winner files (the
    FileCacheManager reads the env at construction, so the swap takes
    effect immediately).  The artifact cache root is untouched.  Study also
    clears the tuners' in-process state (``_reset_inprocess_tuners``): the
    env swap cannot reach winners already held in memory.  The family
    dispatch lives in ``_launch_case_op``; study/production always measure
    (TUNED).
    """
    from cute_gemm.gemm_config import TuneMode

    case_dir = cache_root / _case_slug(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DLKERNEL_CACHE_DIR"] = str(case_dir)
    if mode == "study":
        _reset_inprocess_tuners()

    (a, b, out, kw) = _run_case_tensors(case)
    _launch_case_op(case, a, b, out, kw, TuneMode.TUNED)

    record = {"case": case, "winner_dir": str(case_dir)}
    if out_root is not None and mode == "study":
        record["dataset_path"] = str(_harvest(case_dir, out_root, case))
    return record


def _launch_case_op(case: dict, a, b, out, kw, tune_mode) -> None:
    """The one operator call behind a case, shared by study/production and
    prefill.  ``tune_mode`` is the whole difference: TUNED measures and
    persists (real tensors), STATIC only resolves through the operator's own
    chain (meta tensors cannot be swept).  The GEMM families ride ``gemm``'s
    own kw dispatch (cu_seqlens_m/k); operator-owned families name their
    public entry explicitly.
    """
    family = case["family"]
    if family == "fused_up":
        from cute_gemm.fused_grouped_gemm.interface import moe_up_proj_act

        moe_up_proj_act(a, b, kw["expert_offsets"], tune_mode=tune_mode)
    elif family == "fused_down":
        from cute_gemm.fused_grouped_gemm.interface import moe_down_proj_dact

        # The swiglu preact input rides the tuple's third slot (see builder).
        moe_down_proj_dact(a, b, out, kw["expert_offsets"], tune_mode=tune_mode)
    elif family == "epi":
        _epi_bias_double(a, b, tune_mode=tune_mode, **kw)
    else:
        from cute_gemm.gemm_interface import gemm

        gemm(a, b, out, tune_mode=tune_mode, **kw)


def _expert_offsets(experts: int, tokens: int, device):
    """Expert-major cumulative offsets ([0, ..., tokens], int32) — the
    routed-row contract of the fused families; same construction the varlen
    builders inline.

    Total-preserving rounding: ``(i*T)//E`` is monotone, ends exactly at
    ``tokens``, and per-expert counts differ by at most one — the truncated
    ``arange * (T//E)`` form dropped the remainder (T=123, E=32 covered only
    96 rows; T < E produced an empty group set).
    """
    import torch

    return (torch.arange(experts + 1, dtype=torch.int32, device=device) * tokens) // experts


def _reset_inprocess_tuners() -> None:
    """Clear the in-process tuner winner state (study path only).

    Study must yield one dataset row per raw case, but cases run in one
    process and the tuners hold their winners in memory: two raw cases that
    bucket to the same winner key (e.g. tokens 32767 then 32768, same E)
    would collapse — the second replays the first's in-memory winner, sweeps
    nothing, and writes no winner file into its own ``DLKERNEL_CACHE_DIR``.
    Production must NOT do this: sharing winners across calls is exactly the
    warm-path behavior it deploys.
    """
    from cute_gemm import gemm_interface
    from cute_gemm.gemm_runtime import autotune as epi_autotune

    tuners = [gemm_interface.gemm_tuned, *epi_autotune._MOD_TUNERS.values()]
    for tuner in tuners:
        tuner.cache.clear()
        tuner._measured_contracts.clear()


def _case_slug(case: dict) -> str:
    keys = ("family", "m", "n", "k", "experts", "tokens", "dtype", "seed")
    return "_".join(f"{k}{case[k]}" for k in keys if k in case).replace("/", "-")


def _harvest(case_dir: Path, out_root: Path, case: dict) -> Path:
    """Merge this case's ``.autotune.json`` timings into the dataset file."""
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = out_root / "study_dataset.jsonl"
    rows = []
    for path in sorted(case_dir.rglob("*.autotune.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "case": case,
                "key": payload.get("key"),
                "configs_timings": payload.get("configs_timings"),
                "source": path.name,
            }
        )
    with dataset.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cute_gemm.static_tune")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("study", "production", "prefill"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--out", type=Path, default=Path(".static_tune_out"))
        p.add_argument("--shard", default=None, help="i/N for multi-GPU sharding")
    args = parser.parse_args(argv)

    cache_root = args.out / f"{args.command}_winners"
    if args.command == "prefill":
        return _prefill(args, cache_root)

    if args.command == "study":
        # Prepared-pool sweep: the selector must not prune, and the key's
        # selector-leaf component then reads "fallback" — which is exactly
        # why these winners are data, not deployable entries.
        os.environ["DLKERNEL_VARLEN_SELECTOR"] = "0"
    else:
        os.environ.pop("DLKERNEL_VARLEN_SELECTOR", None)

    records = []
    for case in _iter_cases(args.manifest, args.shard):
        records.append(_run_case(case, args.command, cache_root, args.out))
        print(f"[{args.command}] {_case_slug(case)} done", flush=True)
    (args.out / f"{args.command}_records.json").write_text(json.dumps(records, indent=1))
    return 0


def _prefill(args, cache_root: Path) -> int:
    """Resolve a legal config per case (no measurement), then precompile.

    Uses the same call boundary as runtime so the resolved config matches
    what a real call would pick (GPU-blind boxes run under DLKERNEL_ARCH
    and DLKERNEL_SM_COUNT per plan §3.1).
    """
    from cute_gemm.cache.precompile import precompile

    calls = [_prefill_call(case) for case in _iter_cases(args.manifest, args.shard)]
    report = precompile(calls)
    print(
        f"prefill: {report.n_calls} calls, {report.n_unique} unique, "
        f"{report.n_compiled} compiled here, {report.elapsed_s:.1f}s"
    )
    return 0


def _prefill_call(case: dict):
    """Build the zero-arg call ``precompile`` drives for one case.

    GEMM families keep the rank/pin path: a covered desc pins the config
    (autotuner short-circuit straight to the compile boundary), an uncovered
    desc falls to the operator's own default chain with STATIC.  The
    operator-owned families (fused/epi) skip the ranker entirely — STATIC on
    the public entry lets the operator's own prepare/default chain resolve
    (meta tensors cannot be swept, so nothing is measured either way).
    """
    from cute_gemm.gemm_config import TuneMode
    from cute_gemm.gemm_heuristic import desc_from_call, rank, sm90_candidates

    if case["family"] in _OP_OWNED_FAMILIES:

        def call():
            (a, b, out, kw) = _run_case_tensors_meta(case)
            _launch_case_op(case, a, b, out, kw, TuneMode.STATIC)

        return call

    def call():
        (a, b, out, kw) = _run_case_tensors_meta(case)
        # Pin through the Autotuner entry: a pinned config bypasses
        # auto-selection entirely (autotuner short-circuit), so the meta
        # call runs straight to the compile boundary precompile captures.
        # (The public gemm() variant takes no config= — pinning belongs
        # to this entry.)
        from cute_gemm.gemm_interface import gemm_tuned

        desc = desc_from_call(a, b, out, kw.get("cu_seqlens_m"), kw.get("cu_seqlens_k"))
        config = rank(desc, sm90_candidates(desc))
        if config is not None:
            # Pinned: the autotuner short-circuit runs straight to the
            # compile boundary (no tuning, no disk state).
            gemm_tuned(a, b, out, config=config, **kw)
        else:
            # Uncovered family (dense/gather): the operator's own default
            # chain resolves a legal config (plan §4 "由算子采用合法默认
            # 选择").  STATIC keeps the tuner out (meta tensors cannot be
            # swept) while the -1 path still reaches the boundary.
            gemm_tuned(a, b, out, tune_mode=TuneMode.STATIC, **kw)

    return call


def _run_case_tensors_meta(case: dict):
    """Meta-tensor twin of ``_run_case_tensors`` for ``prefill`` (plan v4 §4).

    ``precompile`` runs each call under ``capture_compiles`` and aborts at the
    first compile boundary -- meta tensors are exactly what that boundary
    expects (real CUDA tensors would allocate, need a GPU, and on a warm
    cache reach no boundary at all, which precompile rejects).  Shapes,
    strides (the varlen-K m-major view) and dtypes match the real builder.
    """
    import torch

    family = case["family"]
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    tokens = case.get("tokens", case.get("m", 4096))
    experts = case.get("experts", 32)
    k, n = case["k"], case["n"]
    if family == "varlen_m":
        a = torch.empty(tokens, k, dtype=dtype, device="meta")
        b = torch.empty(experts, k, n, dtype=dtype, device="meta")
        out = torch.empty(tokens, n, dtype=dtype, device="meta")
        cu = _expert_offsets(experts, tokens, "meta")
        return (a, b, out, {"cu_seqlens_m": cu})
    if family == "varlen_k":
        m = case.get("m", n)
        a = torch.empty(tokens, n, dtype=dtype, device="meta").t()
        b = torch.empty(tokens, m, dtype=dtype, device="meta")
        out = torch.empty(experts, n, m, dtype=dtype, device="meta")
        cu = _expert_offsets(experts, tokens, "meta")
        return (a, b, out, {"cu_seqlens_k": cu})
    if family == "fused_up":
        # moe_up_proj_act: x [tokens, k], gate_up_weight [experts, n, k] with
        # n the literal 2I swiglu concat width; the op allocates (postact,
        # preact) itself.
        a = torch.empty(tokens, k, dtype=dtype, device="meta")
        b = torch.empty(experts, n, k, dtype=dtype, device="meta")
        return (a, b, None, {"expert_offsets": _expert_offsets(experts, tokens, "meta")})
    if family == "fused_down":
        # moe_down_proj_dact: grad_output [tokens, n], down_weight
        # [experts, n, k]; the swiglu preact [tokens, 2k] is an INPUT riding
        # the tuple's third slot (the op allocates its own outputs).
        a = torch.empty(tokens, n, dtype=dtype, device="meta")
        b = torch.empty(experts, n, k, dtype=dtype, device="meta")
        preact = torch.empty(tokens, 2 * k, dtype=dtype, device="meta")
        return (a, b, preact, {"expert_offsets": _expert_offsets(experts, tokens, "meta")})
    if family == "epi":
        # Dense-shaped GEMM plus the fp32 [1, n] row bias; D is allocated by
        # the op (out=None).
        a = torch.empty(case["m"], k, dtype=dtype, device="meta")
        b = torch.empty(k, n, dtype=dtype, device="meta")
        bias = torch.empty(1, n, dtype=torch.float32, device="meta")
        return (a, b, None, {"bias": bias})
    m = case["m"]
    a = torch.empty(m, k, dtype=dtype, device="meta")
    b = torch.empty(k, n, dtype=dtype, device="meta")
    out = torch.empty(m, n, dtype=dtype, device="meta")
    return (a, b, out, {})


def _run_case_tensors(case: dict):
    import torch

    family = case["family"]
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    tokens = case.get("tokens", case.get("m", 4096))
    experts = case.get("experts", 32)
    k, n = case["k"], case["n"]
    torch.manual_seed(case.get("seed", 0))
    if family == "varlen_m":
        a = torch.randn(tokens, k, dtype=dtype, device="cuda")
        b = torch.randn(experts, k, n, dtype=dtype, device="cuda")
        out = torch.empty(tokens, n, dtype=dtype, device="cuda")
        cu = _expert_offsets(experts, tokens, "cuda")
        return (a, b, out, {"cu_seqlens_m": cu})
    if family == "varlen_k":
        # A must be m-major (gemm asserts A.stride(-2) == 1 after the varlen-K
        # relabel): build row-major [tokens, n] and hand back the transposed
        # view.  Found by the pilot study's runner; shapes/dims unchanged.
        a = torch.randn(tokens, n, dtype=dtype, device="cuda").t()
        b = torch.randn(tokens, case.get("m", n), dtype=dtype, device="cuda")
        out = torch.empty(experts, n, case.get("m", n), dtype=dtype, device="cuda")
        cu = _expert_offsets(experts, tokens, "cuda")
        return (a, b, out, {"cu_seqlens_k": cu})
    if family == "fused_up":
        a = torch.randn(tokens, k, dtype=dtype, device="cuda")
        b = torch.randn(experts, n, k, dtype=dtype, device="cuda")
        return (a, b, None, {"expert_offsets": _expert_offsets(experts, tokens, "cuda")})
    if family == "fused_down":
        # Preact input in the tuple's third slot; see the meta twin.
        a = torch.randn(tokens, n, dtype=dtype, device="cuda")
        b = torch.randn(experts, n, k, dtype=dtype, device="cuda")
        preact = torch.randn(tokens, 2 * k, dtype=dtype, device="cuda")
        return (a, b, preact, {"expert_offsets": _expert_offsets(experts, tokens, "cuda")})
    if family == "epi":
        a = torch.randn(case["m"], k, dtype=dtype, device="cuda")
        b = torch.randn(k, n, dtype=dtype, device="cuda")
        bias = torch.randn(1, n, dtype=torch.float32, device="cuda")
        return (a, b, None, {"bias": bias})
    m = case["m"]
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    return (a, b, out, {})


if __name__ == "__main__":
    sys.exit(main())
