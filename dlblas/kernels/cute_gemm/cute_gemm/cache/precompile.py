# Copyright (c) 2026, Tri Dao.
"""Framework-level artifact-cache prewarming: :func:`precompile` / :func:`materialize`.

Two layers, mirroring FlashInfer's ``gen_*_module() -> JitSpec`` /
``build_jit_specs(specs)`` split:

* :class:`cute_gemm.cache.capture.CompileSpec` — one specialization's compile
  request as a self-contained value (identity + how to run it);
* :func:`materialize` — bulk-compile a sequence of specs through the async
  pool (the ``build_jit_specs`` analog);
* :func:`precompile` — the public convenience frontend: drive *calls* under
  capture mode, dedupe the discovered specs by sha, materialize them, and
  return a report.

Discovery calls run ordinary Python up to their first compile boundary; callers
should use meta tensors and avoid device side effects before that boundary.
Compilation happens on CPU workers only — the same tensor-free
``_compile_*`` boundary the pool already uses. The ``.o`` written per spec is
the final loadable artifact (``cute.runtime.load_module`` JITLinks it
directly; there is no separate link step to repeat per process).

Compile counts are **attributed, not diffed**: :func:`materialize` reports
the specializations *this run* compiled — its own in-process jit_cache slow
path plus its pool workers' lock-held compiles (the attribution contract in
:mod:`cute_gemm.cache.async_compile`). Under concurrent multi-process
prewarming each process counts only what it authored, so the per-process
``n_compiled`` values sum to the actual number of compilations, while a
concurrent peer's artifact shows up locally as a hit.

Typical users: a benchmark/validation driver that forks one process per case
(share one ``DLKERNEL_ARTIFACT_CACHE_DIR`` across jobs and precompile the
union once), or a serving process warming shapes before CUDA-graph capture.
Callers that already hold specs in this process can skip the calls and use
:func:`materialize` directly.

Single-threaded callers only (pool + capture-stack contract). Requires the
artifact cache to be enabled (``DLKERNEL_CACHE_ENABLED=1``, the default).
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

from cute_gemm.cache.capture import CompileCaptured, CompileSpec, capture_compiles


class PrecompileReport(NamedTuple):
    """Counts and elapsed time for one :func:`precompile` invocation."""

    n_calls: int
    n_unique: int
    # Compiles *authored by this run*: this process's jit_cache slow path
    # (compiled under the exclusive lock) or this run's pool workers. A
    # concurrent peer's compile is a hit from this process's point of view
    # (we load their artifact), so when several processes prewarm in
    # parallel, their ``n_compiled`` values sum to the actual total number
    # of compilations.
    n_compiled: int
    elapsed_s: float

    @property
    def n_hits(self) -> int:
        """Unique specs loaded without being compiled by this run."""
        return self.n_unique - self.n_compiled


def materialize(specs: Sequence[CompileSpec], stall_timeout_s: float = 600.0) -> int:
    """Ensure every spec's artifact exists on disk; return how many this run compiled.

    Captured specs must still match the current cache root and arguments;
    stale requests are rejected before any compile. Two guarantees make
    "precompile succeeded" mean "artifacts exist":

    * each spec's in-memory cache entry is evicted first, so every call
      exercises the disk path — a warm process cannot satisfy materialization
      from memory (which would leave no artifact), and a corrupt artifact is
      quarantined-and-recompiled instead of being bypassed;
    * after the rotation loop, every spec's ``.o`` must exist — a key whose
      export persistently fails raises instead of reporting success.

    The rotation itself mirrors the autotuner bench loop minus GPU
    benchmarking: submit-on-first-touch, requeue on :class:`CompilePending`,
    sleep when everything is outstanding. A pool failure falls through to an
    in-process compile inside ``jit_cache`` itself, so the loop only needs a
    stall guard against a wedged worker.

    The return value is **exact attribution**, taken from the compile path
    itself rather than diffing artifact existence/mtimes around the loop (a
    diff lets every racing process claim a concurrently-created artifact,
    and misses same-mtime rewrites). A spec counts when this process
    compiled it in-process (the jit_cache slow path's ledger,
    ``jit._LOCAL_COMPILES``) or when this run's pool workers compiled it
    (``CompilePool.compiled_by_pool``). The local ledger is drained at the
    window start so it holds exactly this loop's compiles — including
    recompiles of keys this process compiled on an earlier pass (deleted
    artifacts must re-count), which a snapshot diff would silently drop.
    Finished-but-unobserved worker results are merged before attributing:
    a retry can satisfy itself from the artifact's existence (jit_cache's
    warm fast path checks ``.o`` existence before the pool) without ever
    re-entering ``poll()``. A concurrent peer's compile shows up as a hit —
    the artifact was loaded, not authored here — so across N concurrent
    prewarming processes the returned counts sum to the true total number
    of compilations.

    The per-key exclusive flock normally deduplicates concurrent cold-key
    compiles (see the N-process race test in ``tests/test_cache.py``). After
    a lock timeout, a caller may compile concurrently and attempt a
    no-clobber publish, so compile-once is best effort under contention.
    """
    from cute_gemm.cache import jit as _jit
    from cute_gemm.cache.async_compile import CompilePending, pool_scope
    import cute_gemm.cache as _state

    if not _state.CACHE_ENABLED:
        raise RuntimeError("materialize requires DLKERNEL_CACHE_ENABLED=1")
    for spec in specs:
        if spec.fn.cache_request(*spec.args, **spec.kwargs) != (spec.sha, spec.o_path):
            raise ValueError("stale CompileSpec: cache root, source, or arguments changed")

    # Evict in-memory entries: materialization is a disk contract.
    for spec in specs:
        spec.fn.cache_discard(*spec.args, **spec.kwargs)

    wanted = {spec.sha for spec in specs}
    with pool_scope() as pool:
        submission_marker = pool.n_submitted
        # Open the attribution window: swap out pre-history so the ledger
        # reflects exactly this loop's in-process compiles.
        _jit.drain_local_compiles()
        queue = deque(specs)
        spins = 0
        last_progress = time.monotonic()
        while queue:
            spec = queue.popleft()
            try:
                spec.fn(*spec.args, **spec.kwargs)
                spins = 0
                last_progress = time.monotonic()
            except CompilePending:
                queue.append(spec)
                spins += 1
                if spins >= len(queue):
                    if time.monotonic() - last_progress > stall_timeout_s:
                        raise RuntimeError(
                            f"materialize stalled: no key completed in {stall_timeout_s}s "
                            f"({len(queue)} outstanding)"
                        ) from None
                    time.sleep(0.05)
                    spins = 0
        # Merge worker attributions the rotation loop never observed (a
        # retry that loaded the landed .o via the fast path skipped poll()).
        # poll() on a done future is idempotent; on an unknown sha it is a
        # no-op.
        for spec in specs:
            pool.poll(spec.sha)
        authored = (set(_jit._LOCAL_COMPILES) | pool.compiled_after(submission_marker)) & wanted

    missing = [spec for spec in specs if not Path(spec.o_path).exists()]
    if missing:
        raise RuntimeError(
            f"materialize incomplete: {len(missing)}/{len(specs)} artifacts missing "
            f"after compilation (first sha {missing[0].sha[:12]}); export failures "
            f"were warned above — fix those before relying on the prewarm"
        )
    return len(authored)


def precompile(calls: Sequence[Callable[[], object]]) -> PrecompileReport:
    """Fill the artifact cache with every specialization *calls* would compile.

    Each callable is invoked once under :class:`capture_compiles`; its first
    compile boundary is recorded as a :class:`CompileSpec` and the call
    aborts with :class:`CompileCaptured` (meta-tensor calls cannot proceed
    past compile anyway). Duplicate specializations dedupe by cache sha, then
    all unique specs are materialized in parallel via the pool.

    Callables must be single-kernel in scope (the first boundary aborts the
    call; later boundaries of the same call go undiscovered) and must not be
    needed for their return value. A call that reaches no boundary is rejected
    instead of silently reporting a successful but incomplete prewarm.
    ``n_compiled`` counts the compiles this run authored (exact attribution —
    see :func:`materialize`); warm runs
    report ``n_compiled == 0``, and a concurrent peer's compile counts as a
    hit from this process's point of view.
    """
    import cute_gemm.cache as _state

    if not _state.CACHE_ENABLED:
        raise RuntimeError("precompile requires DLKERNEL_CACHE_ENABLED=1")

    unique: dict = {}
    t0 = time.perf_counter()
    with capture_compiles() as captured:
        for index, call in enumerate(calls):
            try:
                call()
            except CompileCaptured:
                pass
            else:
                raise RuntimeError(f"precompile call {index} did not reach a jit_cache boundary")
    for spec in captured.tasks():
        unique.setdefault(spec.sha, spec)
    specs = list(unique.values())
    n_compiled = materialize(specs)
    return PrecompileReport(
        n_calls=len(calls),
        n_unique=len(specs),
        n_compiled=n_compiled,
        elapsed_s=time.perf_counter() - t0,
    )
