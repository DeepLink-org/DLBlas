# Copyright (c) 2026, Tri Dao.
"""Async kernel compilation: defer-and-retry via a pool of CPU subprocesses.

When a pool is active, ``jit_cache`` handles a ``.o``-cache miss by
submitting the pickled ``(module, qualname, args, kwargs)`` of the
``_compile_*`` function to the pool and raising :class:`CompilePending`
instead of compiling in-process. The caller defers the work item, runs
something else, and retries once the ``.o`` lands (a ~1 ms load). Two
callers implement this loop:

* the pytest plugin's ``--async-compile`` defer loop (tests are the work
  items; see :mod:`cute_gemm.testing.pytest_plugin`);
* the autotuner's bench loop under :func:`pool_scope` (candidate configs
  are the work items; see :class:`cute_gemm.autotuner.Autotuner`).

Design notes:

* **The ``.o`` file is the only rendezvous** between workers and consumers —
  compiled kernels aren't picklable, so the persistent cache doubles as the
  IPC channel, and the per-key ``flock`` in ``jit_cache`` doubles as
  cross-process dedupe (multiple pools / xdist workers coexist safely;
  :func:`_flock_held_exclusively` lets a consumer defer on a key some other
  process is already compiling).
* **Workers never launch kernels by construction**: they call the
  tensor-free ``_compile_*`` functions directly, GPU-blind (arch pinned via
  ``DLKERNEL_ARCH``/``CUTE_DSL_ARCH``, ``CUDA_VISIBLE_DEVICES=""``).
* **Worker startup is an Inductor-style sidecar**: a ``forkserver`` preloads
  torch/cutlass once (:mod:`cute_gemm.cache._pool_preload`, ~13 s) and workers
  fork from it copy-on-write (~0.1 s each). :func:`_neutral_main` keeps
  multiprocessing child prep from re-executing the user's script.
* **Failure semantics**: a failed pool compile is never trusted — the
  consumer falls through to an in-process compile so the real exception
  surfaces with a local traceback.
* **Attribution contract**: ``_pool_worker`` returns
  ``(err, compiled_shas)``. On success the worker drains its in-process
  compile ledger (:func:`cute_gemm.cache.jit.drain_local_compiles`) so the
  result carries exactly which jit-cache keys that worker compiled;
  :class:`CompilePool` accumulates them in ``compiled_by_pool`` as futures
  complete. Callers (e.g. :func:`cute_gemm.cache.precompile.materialize`)
  count authored compiles from these ledgers — never from filesystem
  existence/mtime diffs, which double-count under concurrency.

Env knobs: ``DLKERNEL_ASYNC_COMPILE_START=spawn`` (disable the fork sidecar),
``DLKERNEL_COMPILE_WORKERS`` (shared-executor size, default 16)."""

from __future__ import annotations

import base64
import contextlib
import fcntl
import importlib
import os
import pickle
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple, Optional


class PoolPayload(NamedTuple):
    """Out-of-band worker setup attached to a stable jit-cache argument.

    ``identity`` commits the payload's semantics without putting the generally
    non-deterministic serialized bytes in the persistent cache key. The
    installer must validate it before making the payload visible to the
    compile function.
    """

    installer_module: str
    installer_qualname: str
    identity: str
    data: bytes


def _collect_pool_payloads(obj, out: list[PoolPayload]) -> None:
    """Walk nested tuples collecting ``__dlkernel_pool_payload__`` side-channel
    payloads the worker must install before compiling (e.g. shipping a
    process-local epilogue definition by value). A provider raising means the
    key cannot be shipped — the caller falls back to in-process compile."""
    provider = getattr(obj, "__dlkernel_pool_payload__", None)
    if provider is not None:
        payload = provider()
        if payload is not None:
            if not isinstance(payload, PoolPayload):
                raise TypeError(
                    f"{type(obj).__name__}.__dlkernel_pool_payload__() must return PoolPayload or None"
                )
            out.append(payload)
        return
    if isinstance(obj, tuple):
        for item in obj:
            _collect_pool_payloads(item, out)


def _flock_held_exclusively(lock_path: str) -> bool:
    """True if some process currently holds the flock exclusively.

    Used to detect "another process is compiling this key right now" so the
    consumer defers instead of submitting a duplicate compile to its own
    pool (a duplicate would occupy a pool slot blocked on the same flock).
    """
    try:
        fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except OSError:
            return True
    finally:
        os.close(fd)


class CompilePending(BaseException):
    """A jit_cache miss was submitted to the async compile pool.

    The caller (test) cannot proceed until the ``.o`` exists; the test runner
    should defer the test and retry it later. Carries the cache ``sha`` so
    the runner can poll for completion without re-running the test.

    Derives from :class:`BaseException` (like ``KeyboardInterrupt``) so that
    test-body ``except Exception`` / ``pytest.raises(Exception)`` blocks
    cannot swallow it and turn a not-yet-run test into a false pass. Only
    the plugin's phase hooks are supposed to catch it.
    """

    def __init__(self, sha: str, qualname: str):
        super().__init__(f"kernel compile pending in pool: {qualname} [{sha[:12]}]")
        self.sha = sha
        self.qualname = qualname


def _detect_arch_env() -> tuple[Optional[str], Optional[str]]:
    """Return (DLKERNEL_ARCH, CUTE_DSL_ARCH) for GPU-blind pool workers.

    The two overrides answer different questions. ``DLKERNEL_ARCH`` is the
    Python-side *dispatch* arch (which kernel class / configs get traced) and
    is forwarded verbatim so worker-side dispatch matches the submitter.
    ``CUTE_DSL_ARCH`` is the ptxas *target*, and it must be whatever the main
    process compiles and loads: the explicit env override if set, else the
    physical GPU. It must NOT be derived from ``DLKERNEL_ARCH`` — on the CI
    proxy legs (``DLKERNEL_ARCH=120`` on an H100) the main process still
    compiles the SM120-dispatched code for sm_90a, the only arch the runner
    can load; a worker .o targeting sm_120a would fail cuModuleLoad and
    demote every pool compile to an in-process recompile. Only on a GPU-less
    box (the CPU-only cross-compile workflow) does ``DLKERNEL_ARCH`` double as
    the target. The workers themselves never touch the CUDA driver (no
    context per worker, fork-safe) — this detection runs in the parent.
    """
    dlkernel_arch = os.environ.get("DLKERNEL_ARCH")
    cute_arch = os.environ.get("CUTE_DSL_ARCH")
    physical_arch = None
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            cc = f"{major}{minor}"
            physical_arch = f"sm_{cc}a" if major >= 9 else f"sm_{cc}"
            if cute_arch is None:
                cute_arch = physical_arch
            if dlkernel_arch is None:
                # CUTE_DSL_ARCH may intentionally cross-compile for a target
                # different from the active device; dispatch still follows
                # the physical capability unless explicitly overridden.
                dlkernel_arch = cc
    except Exception:
        pass
    if cute_arch is None and dlkernel_arch is not None:
        # CPU-only box: the dispatch arch is the only target we have.
        from cute_gemm.cute_dsl_utils import _parse_arch_str

        major, minor = _parse_arch_str(dlkernel_arch)
        cc = f"{major}{minor}"
        cute_arch = f"sm_{cc}a" if major >= 9 else f"sm_{cc}"
    if dlkernel_arch is None and cute_arch is not None:
        # No dispatch override: pin workers to the physical arch so their
        # GPU-blind get_device_capacity agrees with the parent's detection.
        dlkernel_arch = cute_arch.removeprefix("sm_").removesuffix("a")
    return dlkernel_arch, cute_arch


def _install_gpu_blind_device_attrs() -> None:
    """Serve the DSL's one TRACE-time driver query from its own static table.

    ``cute.compile`` is driver-free except for one path:
    ``cutlass_dsl.cutlass._generate_kernel_attrs`` calls
    ``cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)`` when the
    launch sets ``min_blocks_per_mp > 1`` without an explicit
    ``preferred_smem_carveout`` (occupancy-bound kernels, e.g. the W4
    small-N decode configs) — in a GPU-blind worker that raises
    ``CUDA_ERROR_NOT_INITIALIZED`` and the key falls back to an in-process
    compile. Answer it from the DSL's static per-arch capacity table
    instead: SM total = per-CTA capacity + the 1 KiB reserved slice
    (233472 on sm_90, verified against the driver), keyed by the pinned
    ``CUTE_DSL_ARCH`` — so worker ``.o`` files stay bit-identical to
    in-process compiles for the target arch. Any other attribute keeps the
    original driver path (fails in the worker, consumer falls back, as
    designed); an arch outside the table raises ValueError, which fails the
    worker the same way."""
    from cutlass.base_dsl.runtime import cuda as cuda_helpers
    from cutlass.utils import get_smem_capacity_in_bytes

    smem_attr = cuda_helpers.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    orig = cuda_helpers.get_device_attribute

    def get_device_attribute(attribute, device_id: int = 0):
        if attribute == smem_attr:
            sm = os.environ.get("CUTE_DSL_ARCH", "").removesuffix("a")
            capacity = get_smem_capacity_in_bytes(sm)
            return capacity + 1024  # per-CTA capacity + reserved = SM total
        return orig(attribute, device_id)

    cuda_helpers.get_device_attribute = get_device_attribute


def _pin_dsl_arch(cute_dsl_arch: Optional[str]) -> None:
    """Force the ptxas target onto the already-built DSL singleton.

    cutlass-dsl >= 4.6.2 snapshots ``CUTE_DSL_ARCH`` when its env manager is
    constructed (4.6.0 read the env lazily) — and construction happens while
    ``cute_gemm``'s import chain pulls in ``cutlass``, before the pool's env
    pinning has run. ``envar.arch`` is the DSL's supported override point;
    without the re-latch, a GPU-blind worker's first ``.arch`` read falls
    back to ``detect_gpu_arch()``'s sm_100a default and every pool ``.o``
    silently targets the wrong arch — cudaErrorNoKernelImageForDevice at
    load time on any other GPU.
    """
    if cute_dsl_arch is None:
        return
    try:
        from cutlass.cutlass_dsl import CuTeDSL

        CuTeDSL._get_dsl().envar.arch = cute_dsl_arch
    except Exception:
        # Singleton not built yet: its constructor reads CUTE_DSL_ARCH from
        # the env (already pinned by the caller), so nothing to re-latch.
        pass


def _pool_initializer(dlkernel_arch: Optional[str], cute_dsl_arch: Optional[str]):
    # GPU-blind compilation: hide devices and pin the target arch via the
    # same overrides the CPU-only compile workflow uses. Forked workers must
    # never initialize CUDA (fork-safety), and spawned workers save the
    # ~1-2 s + ~300 MB of a per-worker CUDA context.
    if dlkernel_arch is not None:
        os.environ["DLKERNEL_ARCH"] = dlkernel_arch
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if cute_dsl_arch is not None:
        os.environ["CUTE_DSL_ARCH"] = cute_dsl_arch
    # Pay the heavy torch/cutlass import at worker start (no-op under
    # forkserver: the preload already imported it before the fork).
    import cute_gemm.cache  # noqa: F401

    # The import above may have constructed the DSL singleton before the env
    # pinning (spawn: no; forkserver: yes, in the sidecar) — re-latch either
    # way, it is idempotent.
    _pin_dsl_arch(cute_dsl_arch)

    if dlkernel_arch is not None:
        # GPU-blind: the driver can never answer, so the one trace-time
        # device query must come from the static arch table.
        _install_gpu_blind_device_attrs()


def _pool_worker(
    mod_name: str,
    qualname: str,
    key_b64: str,
    o_path: str,
    payloads_b64: str = "",
    extra_source_dirs: tuple[str, ...] = (),
) -> tuple[Optional[str], list[str]]:
    """Compile one key.

    Returns ``(err, compiled_shas)``: ``err`` is None on success, else an
    error string (with the worker-side stack tail for worker-only failures).
    ``compiled_shas`` is the worker's in-process compile ledger, drained via
    :func:`cute_gemm.cache.jit.drain_local_compiles` — the exact set of
    jit-cache keys this worker compiled for this task, so the parent can
    attribute produced artifacts to its pool without inferring anything
    from the filesystem. Error paths report no compiled keys.
    """
    drain_local_compiles = None
    original_extra_source_dirs = None
    try:
        from cute_gemm.cache import jit as _jit
        from cute_gemm.cache.jit import drain_local_compiles

        # A long-lived worker handles multiple tasks. A failed prior task may
        # have compiled a nested key before raising; it must not be attributed
        # to this task or any later successful task.
        drain_local_compiles()
        # Honor the submitter's cache root: the jit_cache wrapper recomputes
        # its export path from the live CACHE_DIR, and this worker (forked
        # from a sidecar that snapshotted the env at start) may disagree with
        # a submitter whose CACHE_DIR changed at runtime — the .o would land
        # in the wrong tree and every pool compile would look failed.
        # o_path is <cache_dir>/<source_fingerprint>/<sha>.o.
        import cute_gemm.cache as _state

        _state.CACHE_DIR = os.path.dirname(os.path.dirname(o_path))
        if payloads_b64:
            # Side-channel payloads (``__dlkernel_pool_payload__``): install
            # process-local definitions before the compile fn resolves them.
            for payload in pickle.loads(base64.b64decode(payloads_b64)):
                installer = importlib.import_module(payload.installer_module)
                for part in payload.installer_qualname.split("."):
                    installer = getattr(installer, part)
                installer(payload.identity, payload.data)
        obj = importlib.import_module(mod_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        # The forkserver/spawn worker does not inherit dynamic registrations
        # made in the submitter. Restore them for this task before resolving
        # its fingerprint; a shared worker may serve a different job next.
        original_extra_source_dirs = list(_state.EXTRA_SOURCE_DIRS)
        _state.EXTRA_SOURCE_DIRS[:] = [Path(path) for path in extra_source_dirs]
        _jit._compute_source_fingerprint.cache_clear()
        args, kwargs = pickle.loads(base64.b64decode(key_b64))
        # A long-lived worker may have this key in memory from an earlier
        # cache root or source namespace. It still has to publish this task's
        # requested disk object.
        obj.cache_discard(*args, **kwargs)
        obj(*args, **kwargs)  # jit_cache wrapper: compiles + exports .o
        if not os.path.exists(o_path):
            return "compile succeeded but .o was not exported", []
        return None, drain_local_compiles()
    except Exception as e:
        # The consumer will recompile in-process for a first-class traceback,
        # but a worker-only failure (env/serialization skew) never reproduces
        # there — keep the last frames so those are diagnosable from the
        # stats line alone.
        import traceback

        frames = traceback.format_exception(type(e), e, e.__traceback__)
        tail = "".join(frames[-4:]).strip().replace("\n", " | ")
        return f"{type(e).__name__}: {e} [worker: {tail[-600:]}]", []
    finally:
        if drain_local_compiles is not None:
            drain_local_compiles()
        if original_extra_source_dirs is not None:
            _state.EXTRA_SOURCE_DIRS[:] = original_extra_source_dirs
            _jit._compute_source_fingerprint.cache_clear()


def _make_executor(jobs: int) -> ProcessPoolExecutor:
    """Build a compile-worker executor (Inductor-style forkserver sidecar).

    Forkserver + preload: one sidecar process pays the ~13 s torch/cutlass
    import once, workers fork from it in ~0.1 s each (copy-on-write). The
    forkserver singleton is shared per-process, so multiple executors (the
    test pool, the autotuner's) fork from the same warm sidecar. Opt out
    with DLKERNEL_ASYNC_COMPILE_START=spawn.
    """
    start_method = os.environ.get("DLKERNEL_ASYNC_COMPILE_START", "forkserver")
    ctx = get_context(start_method)
    if start_method == "forkserver":
        ctx.set_forkserver_preload(["cute_gemm.cache._pool_preload"])
    return ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=ctx,
        initializer=_pool_initializer,
        initargs=_detect_arch_env(),
    )


_shared_executor: Optional[ProcessPoolExecutor] = None


def get_shared_executor() -> ProcessPoolExecutor:
    """Executor for ad-hoc compile tasks (e.g. autotuner precompile sweeps).

    Reuses the active :class:`CompilePool`'s executor when one exists (the
    pytest ``--async-compile`` session pool); otherwise lazily creates a
    process-wide executor sized by ``DLKERNEL_COMPILE_WORKERS`` (default 16).    Deliberately ignores :class:`suppress_pool` — suppression turns off the
    *defer-on-miss* behavior of jit_cache, not access to compile workers.
    """
    global _shared_executor
    if _active_pool is not None:
        return _active_pool._executor
    if _shared_executor is None:
        import atexit

        _shared_executor = _make_executor(int(os.environ.get("DLKERNEL_COMPILE_WORKERS", "16")))        # Explicit teardown: without this, the executor is GC'd during
        # interpreter shutdown after its weakref machinery is already gone,
        # printing a spurious "Exception ignored in weakref_cb".
        atexit.register(_shared_executor.shutdown, wait=False, cancel_futures=True)
    return _shared_executor


@contextlib.contextmanager
def _neutral_main():
    """Stop multiprocessing child prep from re-executing the user's script.

    ``Process.start()`` captures preparation data from ``sys.modules['__main__']``:
    for a path-based script the *child* re-runs the whole file via
    ``runpy.run_path`` (so pickles referencing ``__main__`` resolve). Our
    tasks never reference ``__main__`` — they resolve everything by module
    name — and a user script that, say, builds CUDA tensors at import time
    would kill every worker at spawn with "Cannot re-initialize CUDA in
    forked subprocess". Executor workers are spawned synchronously inside
    ``executor.submit`` (``_adjust_process_count``), so masking ``__main__``
    with an empty stub for the duration of the submit is sufficient and
    scoped. Single-threaded callers only (pytest defer loop, autotune bench
    loop).
    """
    import sys
    import types

    real_main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")  # no __file__/__spec__
    try:
        yield
    finally:
        if real_main is not None:
            sys.modules["__main__"] = real_main


class CompilePool:
    """Process pool + in-flight bookkeeping, keyed by jit_cache sha.

    Owns its executor by default; pass ``executor=`` to share one (e.g.
    :func:`pool_scope` wraps the session-long shared executor so scoped
    pools don't respawn workers per autotune sweep). A shared executor is
    not shut down by :meth:`shutdown` — only this pool's futures are
    cancelled.
    """

    def __init__(self, jobs: Optional[int] = None, executor: Optional[ProcessPoolExecutor] = None):
        self._own_executor = executor is None
        self._executor = executor if executor is not None else _make_executor(jobs)
        self._futures: dict[str, Future] = {}
        # Keys being compiled by *another process* (e.g. a different xdist
        # worker's pool), detected via the per-key flock. We defer on them
        # without spending one of our own pool slots on a duplicate compile.
        # sha -> (o_path, lock_path)
        self._external: dict[str, tuple[str, str]] = {}
        self.n_submitted = 0
        # Shas compiled by this pool's workers, accumulated from each
        # finished task's worker-reported compile ledger. Exact attribution
        # of pool-produced artifacts — consumed by precompile.materialize.
        self.compiled_by_pool: set[str] = set()
        # Keep each worker result tied to its submission. Scoped precompile
        # calls may reuse this pool, so its cumulative set alone cannot
        # attribute a compile to the current call.
        self._compiled_by_task: dict[str, set[str]] = {}

    def compiled_after(self, submission_marker: int) -> set[str]:
        """Return compiled keys from tasks submitted after ``submission_marker``.

        Take the marker from ``n_submitted`` before starting a scoped batch.
        A pre-existing future that finishes during that batch belongs to its
        original submitter, even if its result is first polled by the batch.
        """
        authored: set[str] = set()
        for sha in list(self._futures)[submission_marker:]:
            authored.update(self._compiled_by_task.get(sha, ()))
        return authored

    def mark_external(self, sha: str, o_path: str, lock_path: str) -> None:
        """Record that some other process is compiling ``sha`` right now."""
        if sha not in self._futures:
            self._external[sha] = (str(o_path), str(lock_path))

    def prewarm(self) -> None:
        """Start the sidecar + first worker now, off the critical path.

        Same idea as Inductor's ``warm_pool()``: the forkserver's ~13 s
        torch/cutlass preload import starts at the first ``Process`` spawn,
        which is lazy (first submit). Submitting a no-op at session setup
        overlaps that import with pytest collection and the leading warm
        tests instead of the first cold compile.
        """
        with _neutral_main():
            self._executor.submit(os.getpid)

    def submit_raw(
        self, sha: str, mod: str, qualname: str, key_b64: str, o_path: str, payloads_b64: str = ""
    ) -> None:
        if sha in self._futures:
            return
        import cute_gemm.cache as _state

        extra_source_dirs = tuple(str(Path(path).resolve()) for path in _state.EXTRA_SOURCE_DIRS)
        with _neutral_main():
            self._futures[sha] = self._executor.submit(
                _pool_worker, mod, qualname, key_b64, o_path, payloads_b64, extra_source_dirs
            )
        self.n_submitted += 1

    def submit(self, sha: str, fn, args: tuple, kwargs: dict, o_path) -> bool:
        """Submit a live jit_cache miss. Returns False if the key can't be
        shipped to a subprocess (unpicklable args, ``<locals>`` qualname,
        fn defined in ``__main__``, unserializable process-local payloads) —
        the caller should compile in-process instead."""
        if sha in self._futures:
            return True
        if "<locals>" in fn.__qualname__ or fn.__module__ == "__main__":
            # Not resolvable by module+qualname in a worker; compile in-process.
            return False
        try:
            payloads: list = []
            _collect_pool_payloads(args, payloads)
            _collect_pool_payloads(tuple(kwargs.values()), payloads)
            key_b64 = base64.b64encode(pickle.dumps((args, kwargs))).decode("ascii")
            payloads_b64 = (
                base64.b64encode(pickle.dumps(list(dict.fromkeys(payloads)))).decode("ascii")
                if payloads
                else ""
            )
        except Exception:
            return False
        self.submit_raw(sha, fn.__module__, fn.__qualname__, key_b64, str(o_path), payloads_b64)
        return True

    def poll(self, sha: str) -> tuple[str, Optional[str]]:
        """Return (state, error): state in {"new", "pending", "done", "failed"}."""
        fut = self._futures.get(sha)
        if fut is None:
            ext = self._external.get(sha)
            if ext is not None:
                o_path, lock_path = ext
                if os.path.exists(o_path):
                    del self._external[sha]
                    return "done", None
                if _flock_held_exclusively(lock_path):
                    return "pending", None
                # External compiler released the lock without producing a .o
                # (crashed / failed): forget it so the next attempt submits
                # to our own pool.
                del self._external[sha]
            return "new", None
        if not fut.done():
            return "pending", None
        try:
            err, compiled = fut.result()
        except Exception as e:  # BrokenProcessPool etc.
            err, compiled = f"pool worker died: {type(e).__name__}: {e}", []
        self._compiled_by_task[sha] = set(compiled)
        self.compiled_by_pool.update(compiled)
        return ("done", None) if err is None else ("failed", err)

    def stats(self) -> dict:
        done = sum(1 for f in self._futures.values() if f.done())
        errors = []
        for sha, f in self._futures.items():
            if not f.done() or f.cancelled():
                continue
            exc = f.exception()
            err = f"{type(exc).__name__}: {exc}" if exc is not None else f.result()[0]
            if err:
                errors.append((sha, err))
        return {
            "submitted": self.n_submitted,
            "done": done,
            "failed": len(errors),
            "errors": errors,
        }

    def shutdown(self) -> None:
        if self._own_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        else:
            for fut in self._futures.values():
                fut.cancel()


# --- module-level active pool -----------------------------------------------

_active_pool: Optional[CompilePool] = None
_suppress_depth = 0


class suppress_pool:
    """Context manager: make :func:`get_active_pool` return None inside.

    Used by the test runner for a deferred test's final attempt: compile
    in-process (blocking) so a key that never completes in the pool still
    produces a real result or a real traceback instead of deferring forever.
    """

    def __enter__(self):
        global _suppress_depth
        _suppress_depth += 1
        return self

    def __exit__(self, *exc):
        global _suppress_depth
        _suppress_depth -= 1


def activate(jobs: int) -> CompilePool:
    """Activate the session-wide pool (idempotent). Used by the pytest plugin;
    scoped callers should prefer :func:`pool_scope`."""
    global _active_pool
    if _active_pool is None:
        _active_pool = CompilePool(jobs)
    return _active_pool


def deactivate() -> None:
    global _active_pool
    if _active_pool is not None:
        _active_pool.shutdown()
        _active_pool = None


def get_active_pool() -> Optional[CompilePool]:
    return None if _suppress_depth > 0 else _active_pool


@contextlib.contextmanager
def pool_scope():
    """Activate a compile pool for the duration of the block.

    Reuses the globally active pool when one exists (e.g. the pytest
    ``--async-compile`` session pool); otherwise activates a temporary pool
    backed by the shared executor and deactivates it on exit — so
    ``CompilePending`` can only escape into code inside the block, never
    into unrelated user code paths.

    This is how the autotuner overlaps candidate-config compilation with
    benchmarking: the bench loop runs inside ``pool_scope()``, catches
    ``CompilePending`` per config, and retries a config once its ``.o``
    lands (see ``Autotuner.__call__``).
    """
    global _active_pool
    if _active_pool is not None:
        yield _active_pool
        return
    pool = CompilePool(executor=get_shared_executor())
    _active_pool = pool
    try:
        yield pool
    finally:
        _active_pool = None
        pool.shutdown()
