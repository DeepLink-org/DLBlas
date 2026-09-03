"""Low-dependency multi-NPU test for the standalone Torch Planning port.

Run with:
    torchrun --nproc_per_node=${N} -m pytest -s test_planning_triton_ascend_standalone.py

The historical filename is retained for compatibility.  The target package
contains only the Torch implementation: two physical ranks provide local
routing, each rank performs the complete tested Planning arithmetic, and the
helper oracle checks outputs and invariants without CUDA/VMM ABI state.
"""

# =============================================================================
# Environment configuration
# =============================================================================

import os

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), ".triton-cache"),
)


# =============================================================================
# Imports
# =============================================================================

import importlib
import time

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from dlblas.kernels.ascend.moonep.planning_test_utils_ascend import (
    DEDUP_MODES,
    STANDALONE_CASES,
    assert_all_ranks,
    assert_planning_outputs_equal_all_ranks,
    assert_all_ranks_compute,
    case_params,
    dist_env,
    init_case,
    launch_planning_torch_reference,
    make_topk,
    minimal_context_errors,
    planning_invariant_errors,
)


# =============================================================================
# Implementation registration
# =============================================================================

IMPLEMENTATION_MODULES = {
    "torch_reference_port": "dlblas.kernels.ascend.moonep.planning_torch_ascend",
    "triton_reference_semantic": "dlblas.kernels.ascend.moonep.planning_triton_ascend_reference",
}


def _selected_implementations():
    """Import a stable implementation subset shared by every physical rank."""
    requested = os.environ.get(
        "MOONEP_PLANNING_IMPLS", ",".join(IMPLEMENTATION_MODULES)
    )
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = sorted(set(names) - set(IMPLEMENTATION_MODULES))
    if unknown:
        raise ValueError(f"unknown Planning implementations: {unknown}")
    return [importlib.import_module(IMPLEMENTATION_MODULES[name]) for name in names]


def _selected_cases():
    """Select the standalone matrix or a named diagnostic subset."""
    requested = os.environ.get("MOONEP_PLANNING_CASES", "")
    if not requested:
        return STANDALONE_CASES
    names = {name.strip() for name in requested.split(",") if name.strip()}
    selected = [case for case in STANDALONE_CASES if case.name in names]
    if len(selected) != len(names):
        found = {case.name for case in selected}
        raise ValueError(f"unknown standalone Planning cases: {sorted(names - found)}")
    return selected


def _selected_dedup_modes():
    """Select a stable subset of the three public dedup modes."""
    requested = os.environ.get("MOONEP_PLANNING_DEDUP_MODES", ",".join(DEDUP_MODES))
    modes = [mode.strip() for mode in requested.split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(DEDUP_MODES))
    if unknown:
        raise ValueError(f"unknown Planning dedup modes: {unknown}")
    return modes


IMPLEMENTATIONS = _selected_implementations()
TEST_CASES = _selected_cases()
TEST_DEDUP_MODES = _selected_dedup_modes()


# =============================================================================
# Standalone reference contract test
# =============================================================================


@pytest.mark.parametrize("case", case_params(TEST_CASES))
@pytest.mark.parametrize("dedup_mode", TEST_DEDUP_MODES)
def test_standalone_reference_satisfies_invariants(dist_env, case, dedup_mode):
    """Validate the low-dependency oracle and Planning-only invariants."""
    rank, R = dist_env
    ctx = init_case(case, rank, R)
    ctx_errors = minimal_context_errors(ctx)
    assert_all_ranks(
        not ctx_errors,
        rank,
        R,
        f"standalone-reference/{case.name} minimal ctx",
        "; ".join(ctx_errors),
    )
    topk, tokens_per_expert = make_topk(case, rank, R)
    ref_cu_seqlens, ref_plan, ref_src = launch_planning_torch_reference(
        ctx, topk, tokens_per_expert, dedup_mode=dedup_mode
    )
    errors = planning_invariant_errors(
        case,
        ctx,
        topk,
        tokens_per_expert,
        ref_plan,
        ref_cu_seqlens,
        ref_src,
        dedup_mode=dedup_mode,
    )
    assert_all_ranks(
        not errors,
        rank,
        R,
        f"standalone-reference/{case.name}/{dedup_mode} invariants",
        "; ".join(errors[:5]),
    )


# =============================================================================
# Standalone distributed correctness test
# =============================================================================


@pytest.mark.parametrize(
    "implementation",
    IMPLEMENTATIONS,
    ids=lambda module: module.IMPLEMENTATION_NAME,
)
@pytest.mark.parametrize("case", case_params(TEST_CASES))
@pytest.mark.parametrize("dedup_mode", TEST_DEDUP_MODES)
def test_standalone_planning_matches_reference_and_invariants(
    dist_env, case, dedup_mode, implementation
):
    """Run the common ABI, oracle comparisons, and invariants without MoonEP."""
    rank, R = dist_env

    # -------------------------------------------------------------------------
    # Context and physical-rank-local input preparation
    # -------------------------------------------------------------------------
    ctx = init_case(case, rank, R)
    ctx_errors = minimal_context_errors(ctx)
    assert_all_ranks(
        not ctx_errors,
        rank,
        R,
        f"standalone/{case.name} minimal ctx",
        "; ".join(ctx_errors),
    )
    topk, tokens_per_expert = make_topk(case, rank, R)

    # -------------------------------------------------------------------------
    # Tested mode-aware implementation execution and direct return values
    # -------------------------------------------------------------------------
    torch.npu.synchronize()
    started = time.perf_counter()
    cu_seqlens, plan, src = implementation.launch_planning(
        ctx, topk, tokens_per_expert, dedup_mode=dedup_mode
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # -------------------------------------------------------------------------
    # Every-rank Planning launch evidence
    # -------------------------------------------------------------------------
    assert_all_ranks_compute(ctx, rank, R)

    # -------------------------------------------------------------------------
    # Independent standalone reference execution
    # -------------------------------------------------------------------------
    ref_cu_seqlens, ref_plan, ref_src = launch_planning_torch_reference(
        ctx, topk, tokens_per_expert, dedup_mode=dedup_mode
    )

    # -------------------------------------------------------------------------
    # Full public output comparison
    # -------------------------------------------------------------------------
    assert_planning_outputs_equal_all_ranks(
        cu_seqlens,
        plan,
        src,
        ref_cu_seqlens,
        ref_plan,
        ref_src,
        rank,
        R,
        dedup_mode=dedup_mode,
        label_prefix=f"standalone/{implementation.IMPLEMENTATION_NAME}/",
    )

    # -------------------------------------------------------------------------
    # Planning and dedup invariants: no VMM/meta layout state
    # -------------------------------------------------------------------------
    errors = planning_invariant_errors(
        case,
        ctx,
        topk,
        tokens_per_expert,
        plan,
        cu_seqlens,
        src,
        dedup_mode=dedup_mode,
    )
    assert_all_ranks(
        not errors,
        rank,
        R,
        f"standalone/{implementation.IMPLEMENTATION_NAME}/{case.name}/{dedup_mode}",
        "; ".join(errors[:5]),
    )

    # -------------------------------------------------------------------------
    # Comparable end-to-end timing evidence (oracle excluded)
    # -------------------------------------------------------------------------
    if rank == 0:
        print(
            f"[standalone-result] impl={implementation.IMPLEMENTATION_NAME} "
            f"case={case.name} dedup_mode={dedup_mode} "
            f"end_to_end_ms={elapsed_ms:.3f} "
            f"rank_launches={ctx.get('_planning_launch_count', 'missing')}",
            flush=True,
        )
