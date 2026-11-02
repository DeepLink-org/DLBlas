# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Persistent ``.o`` cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files (``.o``) via ``export_to_c``. On
subsequent runs the ``.o`` is loaded via tvm_ffi (~1 ms) instead of
re-generating IR + re-JIT'ing (~500 ms per kernel).

Runtime config (``CACHE_ENABLED``, ``CACHE_DIR``, ``EXTRA_SOURCE_DIRS``)
lives in :mod:`cute_gemm.cache` (the package init).

When an async compile pool is active (see :mod:`cute_gemm.cache.async_compile`),
a cold miss is shipped to a CPU worker and :class:`CompilePending` is raised
instead of compiling in-process; the caller (pytest defer loop, autotune
bench loop) retries once the ``.o`` lands.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import inspect
import os
import pickle
import re
import sys
import tempfile
import time
import warnings
from collections import namedtuple
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

# `cute_gemm.cache` (the package itself) holds the mutable runtime flags as a
# single source of truth; reads happen via attribute access on `_state` so we
# always see the live value, not a snapshot taken at module import.
import cute_gemm.cache as _state  # noqa: E402  (intentional partial-import; see __init__.py)

# Capture mode lives in its own dependency-free module so both the package
# init and this module can reach it without an import cycle.
from cute_gemm.cache import capture as _capture  # noqa: E402


EXPORT_FUNC_NAME = "func"
LOCK_TIMEOUT = 60
# Bump this when the object-loader/worker contract changes.  Old namespaces
# remain on disk but are intentionally unreachable; deleting a user's cache is
# neither necessary nor safe.
_CACHE_FORMAT_VERSION = "jit-object-v2"
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def get_cache_path() -> Path:
    """Return the absolute artifact root so captured paths survive ``chdir``."""
    if _state.CACHE_DIR is not None:
        cache_dir = Path(_state.CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "dlkernel_cache"
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _hash_source_dir(h, root: Path) -> None:
    """Hash all Python sources under *root* into *h*."""
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)


def _normalize_arch(arch: str | None) -> str:
    """Canonicalize an architecture spelling without querying a device."""
    if arch is None:
        return "unknown"
    text = arch.strip().lower()
    match = re.fullmatch(r"(?:sm[_-]?)?(\d+)([af]?)", text)
    if match is None:
        # Keep an unexpected override visible in the namespace rather than
        # silently mapping two distinct values to the same cache directory.
        return text or "unknown"
    return f"sm_{match.group(1)}{match.group(2)}"


def _normalize_dispatch_arch(arch: str | None) -> str:
    """Canonicalize dispatch capability spellings across parent and workers."""
    normalized = _normalize_arch(arch)
    match = re.fullmatch(r"sm_(\d+)([af]?)", normalized)
    if match is not None and not match.group(2) and int(match.group(1)) >= 90:
        # ``get_device_capability`` and DLKERNEL_ARCH expose only major/minor;
        # the ``a`` suffix is a code-generation convention, not a dispatch
        # distinction.  This makes a parent ``sm_90a`` and worker ``90`` share
        # the same identity while keeping CUTE_DSL_ARCH target spelling exact.
        return f"sm_{match.group(1)}a"
    return normalized


def _get_physical_arch() -> str | None:
    """Return the active device capability as host metadata, if available.

    This is intentionally a capability query only.  It does not inspect a
    tensor, transfer data, allocate, or synchronize a stream.  Workers set
    ``CUTE_DSL_ARCH`` before reaching this helper, so GPU-blind compilation
    does not need a CUDA context.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    suffix = "a" if major >= 9 else ""
    return f"sm_{major}{minor}{suffix}"


def _get_torch_abi() -> tuple[str, str]:
    """Return the PyTorch/CUDA ABI labels without touching tensor data."""
    try:
        import torch

        return str(torch.version.cuda or "none"), str(torch.__version__)
    except Exception:
        return "unknown", "unknown"


def _cache_environment_identity(
    *,
    cute_dsl_arch: str | None = None,
    dlkernel_arch: str | None = None,
    physical_arch: str | None = None,
    cuda_version: str | None = None,
    torch_version: str | None = None,
) -> str:
    """Describe only inputs that can change whether a cached object executes.

    ``CUTE_DSL_ARCH`` is the code-generation target and ``DLKERNEL_ARCH`` is
    the Python dispatch target.  A physical GPU ordinal is deliberately absent:
    ``CUDA_VISIBLE_DEVICES`` may remap equivalent devices without changing the
    generated code.  Optional arguments make this function deterministic and
    easy to test without a CUDA runtime.
    """
    if cute_dsl_arch:
        target = _normalize_arch(cute_dsl_arch)
    elif physical_arch:
        target = _normalize_arch(physical_arch)
    else:
        # CPU-only compilation derives its ptxas target from DLKERNEL_ARCH,
        # matching async_compile._detect_arch_env's fallback.
        target = _normalize_dispatch_arch(dlkernel_arch)
    dispatch_source = dlkernel_arch or physical_arch or cute_dsl_arch
    dispatch = _normalize_dispatch_arch(dispatch_source)
    return ";".join(
        (
            f"format={_CACHE_FORMAT_VERSION}",
            f"target={target}",
            f"dispatch={dispatch}",
            f"cuda={cuda_version or 'unknown'}",
            f"torch={torch_version or 'unknown'}",
        )
    )


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash cute_gemm + extra source dirs plus runtime ABI stamps into a fingerprint."""
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    physical_arch = _get_physical_arch()
    cuda_version, torch_version = _get_torch_abi()
    h.update(
        _cache_environment_identity(
            cute_dsl_arch=os.environ.get("CUTE_DSL_ARCH"),
            dlkernel_arch=os.environ.get("DLKERNEL_ARCH"),
            physical_arch=physical_arch,
            cuda_version=cuda_version,
            torch_version=torch_version,
        ).encode()
    )
    # Hash the entire `cute_gemm` package, not just `cute_gemm/cache/`. Resolving via
    # the top-level package import keeps the fingerprint stable regardless of
    # where inside the package this file lives.
    import cute_gemm as _pkg

    _hash_source_dir(h, Path(_pkg.__file__).resolve().parent)
    for extra_dir in _state.EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())
    return h.hexdigest()


def _key_to_hash(key: tuple) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _fsync_path(path: Path, *, is_dir: bool = False) -> None:
    """Best-effort fsync for cross-node visibility (DeepGEMM-style hardening).

    Renames alone don't guarantee another host sees the data on every
    distributed filesystem; flushing the file before publish and the parent
    directory after closes that gap. Failures on exotic filesystems degrade
    to plain-rename semantics rather than aborting the publish.
    """
    flags = os.O_RDONLY | (os.O_DIRECTORY if is_dir else 0)
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _publish_artifact(tmp_path: Path, o_path: Path) -> bool:
    """Atomically publish a freshly exported artifact; True if we won.

    DeepGEMM's directory-rename protocol, file edition: correctness comes
    from filesystem atomicity alone, never from the advisory lock. The
    no-clobber ``os.link`` is the single publish primitive — EEXIST means
    some other process/rank/host already published this key, and the loser
    discards its own export and reuses the existing artifact. Hardlinks are
    atomic on POSIX and NFS. Any other OSError propagates to the caller's
    export-failure handling (warn + keep the in-process result).
    """
    _fsync_path(tmp_path)
    try:
        os.link(tmp_path, o_path)
        won = True
    except FileExistsError:
        won = False  # another publisher won the race; reuse its artifact
    _fsync_path(o_path.parent, is_dir=True)
    tmp_path.unlink(missing_ok=True)
    return won


def _export_artifact(compiled_fn, o_path: Path) -> None:
    """Export through a unique private file, including when a lock times out.

    A PID-only name is unsafe when two threads compile the same key: the
    fallback exporter could overwrite the active writer's file before it is
    hard-linked into the cache. ``mkstemp`` reserves a distinct name for each
    export; CuTe's exporter opens the path in write mode and replaces its bytes.
    """
    fd, name = tempfile.mkstemp(prefix=f"{o_path.name}.tmp.", dir=o_path.parent)
    os.close(fd)
    tmp_path = Path(name)
    try:
        compiled_fn.export_to_c(
            object_file_path=str(tmp_path),
            function_name=EXPORT_FUNC_NAME,
        )
        _publish_artifact(tmp_path, o_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _file_digest(path: str) -> str:
    """Digest source bytes, including edits that preserve size and mtime."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        # A missing source cannot contribute its bytes to the identity.
        return "nosrc"


def _module_source_digest(fn) -> str:
    """Digest of the source file defining *fn*'s module.

    Folds downstream-module edits into the key even when the
    module's source dir is not registered in ``EXTRA_SOURCE_DIRS`` (the
    package-wide fingerprint only covers registered trees). Synthetic
    modules (exec'd strings, REPL definitions) have no source file and get
    the constant ``"nosrc"``.
    """
    try:
        src = inspect.getsourcefile(fn)
    except (OSError, TypeError):
        return "nosrc"
    if src is None:
        return "nosrc"
    return _file_digest(src)


def _request_sha(fn, cache_key: tuple, source_digest: str | None = None) -> str:
    """Disk sha over (module, qualname, module-source digest, args).

    The module is part of the identity: two modules may define same-named
    compile functions with identical signatures (verified 2026-09-23 —
    without it, the second module silently loaded the first's artifact).
    Pool workers already resolve tasks by module + qualname, so this makes
    the on-disk key match the dispatch identity exactly. The source
    fingerprint covers the cute_gemm package plus ``EXTRA_SOURCE_DIRS``; the
    per-module digest additionally invalidates keys when a *downstream*
    module that never registered its source dir edits its ``_compile_*``
    definitions (previously such edits kept serving stale artifacts).
    The wrapper freezes the defining source digest when the function is
    decorated. An old function left alive after an on-disk edit must never
    export its old code under the new source's key.
    """
    return _key_to_hash(
        (fn.__module__, fn.__qualname__, source_digest or _module_source_digest(fn)) + cache_key
    )


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory file lock using fcntl.flock with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# In-process compile attribution
# ---------------------------------------------------------------------------

# Shas compiled by *this process* at the two real in-process compile points
# of the jit_cache wrapper below (slow-path-under-exclusive-lock and the
# lock-timeout fallback). Answering "which compiles did this run author"
# from the compile path itself — instead of diffing artifact existence or
# mtimes around a call — stays correct under concurrency: every racing
# process attributes exactly the keys it compiled, never a peer's artifact
# (mtime-equal rewrites included). Loads and pool submissions deliberately
# do NOT record here: they don't compile.
_LOCAL_COMPILES: set[str] = set()


def note_local_compile(sha: str) -> None:
    """Record that this process just compiled key ``sha``."""
    _LOCAL_COMPILES.add(sha)


def drain_local_compiles() -> list[str]:
    """Swap-out: return the locally compiled shas so far and clear the ledger.

    Pool workers drain this after each task so their result can carry the
    exact set of keys they compiled back to the parent process.
    """
    shas = list(_LOCAL_COMPILES)
    _LOCAL_COMPILES.clear()
    return shas


# ---------------------------------------------------------------------------
# JIT cache decorator
# ---------------------------------------------------------------------------


def jit_cache(fn):
    """Decorator that caches compiled CuTe DSL kernels in-memory and on disk.

    The decorated function should return a compiled kernel (i.e. call cute.compile).
    The disk key includes module identity, frozen defining source and arguments.

    Concurrency model
    -----------------
    The disk side uses a per-key ``{sha}.lock`` file (advisory ``flock``):

    * **Fast path (warm cache).** If the ``.o`` file already exists, we take a
      shared lock just long enough to ``load_module`` it. Many readers can
      proceed concurrently.
    * **Slow path (cold cache).** Normally the actual ``fn(*args, **kwargs)``
      compile runs *under* the exclusive lock. This serializes redundant
      compilations of the same key across xdist workers / processes. After a
      lock timeout, a caller may compile concurrently and attempt a no-clobber
      publish, so compile-once is a best-effort property under contention.
      Previously the compile ran *between* the shared-lock check and the
      exclusive-lock export, so all
      N processes wasted CPU compiling the same key in parallel — wall time
      was unchanged but compile-CPU pressure scaled with concurrency, which
      starved other compiles when many keys were cold at once.)

    The lock is per-key, so distinct keys never contend with each other.
    """
    cache = {}
    hits = 0
    misses = 0
    source_digest = _module_source_digest(fn)

    def _memory_key(args, kwargs):
        return args + tuple(sorted(kwargs.items())) if kwargs else args

    def cache_request(*args, **kwargs):
        """Resolve this wrapper's current persistent request identity."""
        sha = _request_sha(fn, _memory_key(args, kwargs), source_digest)
        o_path = get_cache_path() / _compute_source_fingerprint() / f"{sha}.o"
        return sha, o_path

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = _memory_key(args, kwargs)

        # Snapshot once per call so a concurrent flip of ``_state.CACHE_ENABLED``
        # mid-call can't desync the disk-path branch.
        enabled = _state.CACHE_ENABLED

        # 0. Capture mode (cute_gemm.cache.capture): record the boundary and
        #    abort the call instead of compiling/loading. Discovery-only —
        #    callers drive entry points with meta tensors and expect the
        #    abort. Runs BEFORE the in-memory check so discovery reports the
        #    workload's full key set even for keys this process already holds
        #    (a fresh process would capture them). The task stores the
        #    *wrapper* (this function), not the raw ``fn``, so replays go
        #    through the full cache path (miss counters, pool submission,
        #    export) instead of silently bypassing it. Requires the object
        #    cache to be enabled — capture IS cache-key discovery.
        if enabled and _capture.stack.frames:
            sha, capture_o_path = cache_request(*args, **kwargs)
            _capture.stack.record(_capture.CompileSpec(sha, wrapper, args, kwargs, capture_o_path))
            raise _capture.CompileCaptured(sha, fn.__qualname__)

        # 1. In-memory hit. Same process already compiled or loaded this key.
        if cache_key in cache:
            hits += 1
            return cache[cache_key]

        # 2. Cache disabled: pure in-process compile, no disk side effects.
        if not enabled:
            misses += 1
            compiled_fn = fn(*args, **kwargs)
            cache[cache_key] = compiled_fn
            return compiled_fn

        sha, o_path = cache_request(*args, **kwargs)
        cache_path = o_path.parent
        cache_path.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path / f"{sha}.lock"

        def _load_cached() -> object:
            """Load the .o into a callable; caller guarantees existence."""
            m = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
            return m[EXPORT_FUNC_NAME]

        def _quarantine_corrupt(exc: Exception) -> None:
            """A cached .o that fails to load (truncated write from a killed
            worker, missing __tvm_ffi_func, ...) is a cache miss, not an error:
            delete it so this and future processes recompile instead of failing
            forever (the CI cache persists across runs)."""
            warnings.warn(
                f"cute_gemm cache: corrupt cached object for key {sha} "
                f"({type(exc).__name__}: {exc}); deleting and recompiling",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                o_path.unlink()
            except OSError:
                pass

        # 3. Fast path: optimistic existence check, then shared-lock load.
        #    The unlocked ``.exists()`` is a no-cost short-circuit for warm
        #    caches; the shared lock guards against reading a partial file
        #    while a concurrent writer holds the exclusive lock.
        if o_path.exists():
            try:
                with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                    if o_path.exists():
                        try:
                            loaded = _load_cached()
                        except Exception:
                            # A shared reader cannot unlink the entry: another
                            # shared reader may still be loading it. The
                            # exclusive slow path rechecks and repairs it.
                            pass
                        else:
                            cache[cache_key] = loaded
                            hits += 1
                            return loaded
            except RuntimeError:
                pass  # lock timeout; fall through to slow path

        # 3b. Async-compile pool: on a cold miss with a pool
        #     active, ship the key to a CPU subprocess and raise
        #     CompilePending instead of compiling in-process. The test runner
        #     defers the test and retries once the worker has exported the
        #     .o. Pool failures fall through to the in-process compile below
        #     so the real exception surfaces with a local traceback.
        from cute_gemm.cache import async_compile as _async

        pool = _async.get_active_pool()
        if pool is not None:
            state, err = pool.poll(sha)
            if state == "new":
                # If another process (e.g. a different xdist worker's pool)
                # holds the exclusive per-key flock, it is compiling this key
                # right now: defer on it instead of submitting a duplicate.
                if _async._flock_held_exclusively(str(lock_path)):
                    pool.mark_external(sha, str(o_path), str(lock_path))
                    raise _async.CompilePending(sha, fn.__qualname__)
                if pool.submit(sha, fn, args, kwargs, o_path):
                    raise _async.CompilePending(sha, fn.__qualname__)
                # unpicklable key / <locals> qualname: compile in-process
            elif state == "pending":
                raise _async.CompilePending(sha, fn.__qualname__)
            elif state == "done":
                try:
                    with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                        if o_path.exists():
                            try:
                                loaded = _load_cached()
                            except Exception:
                                # Repair only after the exclusive lock below.
                                pass
                            else:
                                cache[cache_key] = loaded
                                hits += 1
                                return loaded
                except RuntimeError:
                    pass  # lock timeout; fall through to slow path
            else:  # "failed"
                # warnings.warn, not print: this fires inside a test whose stdout
                # pytest captures and discards on pass (the in-process fallback
                # usually succeeds), so a print never reaches the user — the
                # warning lands in pytest's warnings summary instead.
                warnings.warn(
                    f"cute_gemm cache: async compile failed for {fn.__qualname__} "
                    f"[{sha[:12]}]: {err}; recompiling in-process for a real traceback",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # 4. Slow path: take EXCLUSIVE lock and compile under it. The recheck
        #    inside the lock catches the race where another process compiled
        #    while we were waiting; in that case we just load and return
        #    without duplicating the compile.
        try:
            lock = FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT)
            # Acquire outside the compile's try scope: only the acquisition
            # raises the timeout RuntimeError. A blanket `try: with lock: fn()`
            # also caught RuntimeErrors from the compile itself, mislabeling
            # real compile failures as lock timeouts and re-running the failed
            # compile a second time.
            lock.__enter__()
        except RuntimeError as e:
            # Lock acquisition timed out (heavy contention, stuck holder, or
            # a filesystem where flock semantics don't hold). DeepGEMM-style
            # degraded path: compile unlocked and publish no-clobber — if a
            # peer already published this key we simply lose the race and
            # keep our in-process result; if nobody did, the artifact is
            # banked for everyone. Correctness never depended on the lock.
            warnings.warn(
                f"cute_gemm cache: lock timeout for key {sha}: {e}; "
                f"falling back to unlocked compile + no-clobber publish",
                RuntimeWarning,
                stacklevel=2,
            )
            misses += 1
            compiled_fn = fn(*args, **kwargs)
            note_local_compile(sha)
            try:
                _export_artifact(compiled_fn, o_path)
            except Exception as export_err:
                warnings.warn(
                    f"cute_gemm cache: export failed for key {sha}: {export_err} "
                    f"(this key will recompile every run)",
                    RuntimeWarning,
                    stacklevel=2,
                )
            cache[cache_key] = compiled_fn
            return compiled_fn
        try:
            if o_path.exists():
                try:
                    loaded = _load_cached()
                except Exception as e:
                    _quarantine_corrupt(e)  # holds the exclusive lock: safe
                else:
                    cache[cache_key] = loaded
                    hits += 1
                    return loaded

            misses += 1
            compiled_fn = fn(*args, **kwargs)
            note_local_compile(sha)
            # Export to a private temp file, then publish atomically:
            # correctness never depends on the advisory lock. A process
            # killed mid-export (xdist worker OOM-kill, timeout) cannot
            # leave a truncated .o at the final path, and a concurrent
            # publisher on a host where flock semantics don't hold simply
            # loses the no-clobber race and loads the winner's artifact.
            try:
                _export_artifact(compiled_fn, o_path)
            except Exception as e:
                warnings.warn(
                    f"cute_gemm cache: export failed for key {sha}: {e} "
                    f"(this key will recompile every run)",
                    RuntimeWarning,
                    stacklevel=2,
                )
            cache[cache_key] = compiled_fn
            return compiled_fn
        finally:
            lock.__exit__(None, None, None)

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_discard(*args, **kwargs):
        """Evict one in-process entry so a materializer exercises disk state."""
        cache.pop(_memory_key(args, kwargs), None)

    def cache_info():
        return CacheInfo(hits=hits, misses=misses, maxsize=None, currsize=len(cache))

    wrapper.cache = cache
    wrapper.cache_request = cache_request
    wrapper.cache_discard = cache_discard
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper
