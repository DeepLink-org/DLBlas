# Copyright (c) 2026, Tri Dao.
"""Capture mode for :func:`cute_gemm.cache.jit_cache` — discovery without compiling.

``capture_compiles()`` turns every ``jit_cache``-decorated compile boundary
into a recorded :class:`CompileSpec` instead of a compile/load: the wrapper
stores the spec and raises :class:`CompileCaptured` to abort the enclosing
call. That abort is the point — callers drive operator entry points purely
to discover which specializations a workload needs, and a meta-tensor call
cannot proceed past the compile boundary anyway.

This is the discovery half of :func:`cute_gemm.cache.precompile.precompile`;
it works for every ``@jit_cache`` kernel family with no per-operator code.
The *stable contract* is "the capture hook records compilation requests";
driving calls with meta tensors is one frontend for producing those calls,
not a requirement on operator families.

Single-threaded callers only (same contract as the async compile pool): the
capture stack is process-global state consulted on the jit_cache path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, NamedTuple, Optional


class CompileSpec(NamedTuple):
    """One specialization's process-local compile request.

    Mirrors FlashInfer's ``JitSpec`` role: a *description* of a compilation
    (identity + how to run it), independent of any particular call that
    produced it. Specs can be deduped by ``sha`` and materialized later in
    the same process via :func:`cute_gemm.cache.precompile.materialize` without
    re-driving the original call. The callable and cache path make this
    unsuitable as a serialized, portable manifest.

    ``fn`` is the jit_cache **wrapper** (not the raw compile function), so
    materialization goes through the full cache path: miss counters, pool
    submission, and artifact export.
    """

    sha: str
    fn: Callable
    args: tuple
    kwargs: dict
    o_path: Path


class CompileCaptured(BaseException):
    """A compile boundary fired while capture mode was active.

    Derives from :class:`BaseException` (like :class:`CompilePending`) so a
    test-body ``except Exception`` cannot swallow the abort and let the call
    continue into a meta-tensor launch. Callers doing discovery catch it
    explicitly; everyone else lets it propagate.
    """

    def __init__(self, sha: str, qualname: str):
        super().__init__(f"compile boundary captured: {qualname} [{sha[:12]}]")
        self.sha = sha
        self.qualname = qualname


class _CaptureStack:
    """Registry of captured compile tasks (module-level singleton).

    ``frames`` is a plain list read directly (list truthiness) on the
    jit_cache call path — the check must cost nothing when capture is off.
    """

    def __init__(self) -> None:
        self.frames: List[dict] = []

    def record(self, spec: CompileSpec) -> None:
        """Record one compile boundary hit. Called from the jit_cache wrapper."""
        self.frames[-1][spec.sha] = spec

    def push(self) -> dict:
        frame: dict = {}
        self.frames.append(frame)
        return frame

    def pop(self, frame: dict) -> None:
        top = self.frames.pop()
        assert top is frame, "capture stack corrupted (unbalanced push/pop)"


#: Module-level capture state; ``cute_gemm.cache.jit`` reads this once per
#: compile-boundary call (one attribute check on the fully-warm path).
stack = _CaptureStack()


class capture_compiles:
    """Context manager: record jit_cache compile boundaries instead of running them.

    >>> with capture_compiles() as captured:
    ...     try:
    ...         gemm(a_meta, b_meta, cu_seqlens_m=cu_meta)
    ...     except CompileCaptured:
    ...         pass
    ...     captured.tasks()  # [CompileSpec(...), ...]

    Reentrant: nesting captures independently. The captured mapping is
    sha-keyed, so a call sequence that repeats a specialization records it
    once.
    """

    def __init__(self) -> None:
        self._frame: Optional[dict] = None

    def __enter__(self) -> "capture_compiles":
        self._frame = stack.push()
        return self

    def __exit__(self, *exc) -> None:
        stack.pop(self._frame)

    def tasks(self) -> List[CompileSpec]:
        """Captured specs, arbitrary-but-stable order.

        Valid inside and after the context (the frame is immutable once
        popped), so callers can inspect the harvest after the ``with`` block.
        """
        assert self._frame is not None, "tasks() called before entering the context"
        return list(self._frame.values())
