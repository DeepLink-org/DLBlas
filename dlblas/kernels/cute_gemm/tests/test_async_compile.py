# Copyright (c) 2026, Tri Dao.
"""Regression tests for the async compile pool (``--async-compile``).

Each test encodes a specific failure mode discovered while building the
feature — see cute_gemm/cache/async_compile.py and the defer-and-retry loop in
cute_gemm/testing/pytest_plugin.py. The CPU-only worker/pool-scope tests below
were merged from test_async_compile_stress.py (placeholder object files: pool
bookkeeping and the compile/export protocol, not CUDA code generation).
"""

import base64
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path

import pytest

import cute_gemm.cache as cache_state
from cute_gemm.cache import async_compile as ac
from cute_gemm.cache import jit as jit_module
from cute_gemm.cache import jit_cache
from cute_gemm.cache.async_compile import CompilePending, CompilePool, _flock_held_exclusively


# ---------------------------------------------------------------------------
# CompilePending semantics
# ---------------------------------------------------------------------------


def test_compile_pending_is_base_exception():
    """CompilePending must NOT be catchable by ``except Exception``.

    Regression: test bodies with broad ``try/except Exception`` (or
    ``pytest.raises(Exception)``) around a kernel call would otherwise
    swallow the deferral signal and turn a never-executed test into a false
    pass.
    """
    assert issubclass(CompilePending, BaseException)
    assert not issubclass(CompilePending, Exception)

    with pytest.raises(BaseException):
        try:
            raise CompilePending("0" * 64, "some._compile_fn")
        except Exception:  # must NOT catch it
            pytest.fail("except Exception swallowed CompilePending")

    exc = CompilePending("ab" * 32, "cute_gemm.foo._compile_bar")
    assert exc.sha == "ab" * 32
    assert "cute_gemm.foo._compile_bar" in str(exc)


# ---------------------------------------------------------------------------
# Pool bookkeeping: poll states, external-compile detection
# ---------------------------------------------------------------------------


def test_flock_probe_detects_exclusive_holder(tmp_path):
    """The external-compile probe: exclusive flock held => True, free => False.

    This is what lets one xdist worker defer on a key that another worker's
    pool is already compiling, instead of burning a pool slot on a duplicate
    compile that would just block on the same flock.
    """
    import fcntl

    lock_path = tmp_path / "key.lock"
    assert not _flock_held_exclusively(str(lock_path))  # nonexistent -> free

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert _flock_held_exclusively(str(lock_path))
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert not _flock_held_exclusively(str(lock_path))
    finally:
        os.close(fd)


def test_pool_poll_external_lifecycle(tmp_path):
    """poll() on an externally-compiled key: pending while locked, done once
    the .o appears, back to new if the external compiler died lockless.

    Regression: without the died-without-.o transition, a crashed foreign
    compiler would leave every waiter deferring forever (until the wedge
    deadline force-synced them one by one).
    """
    import fcntl

    pool = CompilePool.__new__(CompilePool)  # skip executor spawn: bookkeeping only
    pool._futures = {}
    pool._external = {}
    pool.n_submitted = 0

    sha = "e" * 64
    o_path = tmp_path / f"{sha}.o"
    lock_path = tmp_path / f"{sha}.lock"

    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        pool.mark_external(sha, str(o_path), str(lock_path))
        assert pool.poll(sha) == ("pending", None)  # locked, no .o yet

        o_path.write_bytes(b"fake object file")
        assert pool.poll(sha) == ("done", None)  # .o landed
        assert sha not in pool._external  # consumed

        # Crash case: locked then released without producing a .o.
        sha2 = "f" * 64
        o2 = tmp_path / f"{sha2}.o"
        pool.mark_external(sha2, str(o2), str(lock_path))
        assert pool.poll(sha2) == ("pending", None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert pool.poll(sha2) == ("new", None)  # forgotten -> resubmittable
        assert sha2 not in pool._external
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Worker requests, source namespaces, and pool scoping (CPU-only)
# ---------------------------------------------------------------------------


class _Compiled:
    def export_to_c(self, object_file_path, function_name):
        Path(object_file_path).write_bytes(b"async-stress-stub")


class _ExportFails:
    def export_to_c(self, object_file_path, function_name):
        raise OSError("simulated full artifact filesystem")


@jit_cache
def _nested_compile(key):
    return _Compiled()


@jit_cache
def _compile_then_fail(key):
    _nested_compile(key)
    raise RuntimeError("outer compilation failed after an inner compile")


@jit_cache
def _compile_export_failure(key):
    return _ExportFails()


@jit_cache
def _healthy_compile(key):
    return _Compiled()


def _request(fn, key, root):
    """Build the same request/path that jit_cache sends to a pool worker."""
    sha = jit_module._request_sha(fn.__wrapped__, (key,))
    o_path = root / jit_module._compute_source_fingerprint() / f"{sha}.o"
    encoded = base64.b64encode(pickle.dumps(((key,), {}))).decode("ascii")
    return sha, encoded, str(o_path)


@pytest.fixture
def isolated_worker_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_state, "CACHE_ENABLED", True)
    monkeypatch.setattr(cache_state, "CACHE_DIR", str(tmp_path / "artifacts"))
    jit_module.drain_local_compiles()
    for fn in (_nested_compile, _compile_then_fail, _compile_export_failure, _healthy_compile):
        fn.cache_clear()
    # The pool-shippable conftest stub is module-level, so its in-memory
    # cache survives across tests: clear it or a later test reusing the same
    # key under a different CACHE_DIR silently memory-hits instead of
    # exercising the disk path.
    import conftest as _conftest

    _conftest.compile_stub.cache_clear()
    yield tmp_path / "artifacts"
    jit_module.drain_local_compiles()
    _conftest.compile_stub.cache_clear()


@pytest.mark.parametrize("failure", ["nested", "export"])
def test_failed_worker_request_does_not_attribute_old_compiles(isolated_worker_cache, failure):
    """A reused worker must not credit a later task with an earlier failed task's SHA.

    A failure can happen after an inner compile or while exporting the outer
    object (for example, a full cache filesystem). Both paths leave a real
    compile in the worker before the task reports an error.
    """
    root = isolated_worker_cache
    failing_fn = _compile_then_fail if failure == "nested" else _compile_export_failure
    stale_fn = _nested_compile if failure == "nested" else _compile_export_failure
    bad_sha, bad_args, bad_path = _request(failing_fn, failure, root)
    stale_sha, _, _ = _request(stale_fn, failure, root)

    with pytest.warns(RuntimeWarning) if failure == "export" else nullcontext():
        err, compiled = ac._pool_worker(__name__, failing_fn.__qualname__, bad_args, bad_path)
    assert err is not None
    assert compiled == []

    good_sha, good_args, good_path = _request(_healthy_compile, failure, root)
    err, compiled = ac._pool_worker(__name__, _healthy_compile.__qualname__, good_args, good_path)
    assert err is None
    assert Path(good_path).read_bytes() == b"async-stress-stub"
    assert set(compiled) == {good_sha}
    assert stale_sha not in compiled
    assert bad_sha not in compiled


def test_worker_places_successive_requests_in_their_submitter_roots(
    isolated_worker_cache, tmp_path
):
    """A sidecar worker reused across jobs must honor each job's cache directory."""
    root_a = isolated_worker_cache
    root_b = tmp_path / "other-job-artifacts"
    for key, root in (("job-a", root_a), ("job-b", root_b)):
        sha, args, path = _request(_healthy_compile, key, root)
        err, compiled = ac._pool_worker(__name__, _healthy_compile.__qualname__, args, path)
        assert err is None
        assert set(compiled) == {sha}
        assert Path(path).read_bytes() == b"async-stress-stub"
    assert len(list(root_a.rglob("*.o"))) == 1
    assert len(list(root_b.rglob("*.o"))) == 1


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_real_worker_uses_submitter_extra_source_dirs(
    isolated_worker_cache, monkeypatch, tmp_path, start_method
):
    """A worker's object must land in the submitter's source-fingerprint tree."""
    tests_dir = str(Path(__file__).resolve().parent)
    monkeypatch.syspath_prepend(tests_dir)
    import conftest as stub

    helper_dir = tmp_path / "external-helper"
    helper_dir.mkdir()
    (helper_dir / "helper.py").write_text("CODEGEN_VERSION = 1\n")
    monkeypatch.setattr(cache_state, "EXTRA_SOURCE_DIRS", [helper_dir])
    monkeypatch.setenv("DLKERNEL_ASYNC_COMPILE_START", start_method)
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    jit_module._compute_source_fingerprint.cache_clear()
    root = isolated_worker_cache
    pool = ac.CompilePool(jobs=1)
    try:
        pool.prewarm()
        sha, _, path = _request(stub.compile_stub, "external-helper", root)
        assert pool.submit(sha, stub.compile_stub.__wrapped__, ("external-helper",), {}, path)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state, err = pool.poll(sha)
            assert state != "failed", err
            if state == "done":
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker did not finish")
        assert Path(path).read_bytes() == b"pool-shipped-stub"
        assert {str(p) for p in root.rglob("*.o")} == {path}
    finally:
        pool.shutdown()
        jit_module._compute_source_fingerprint.cache_clear()


def test_reused_worker_materializes_same_key_in_new_source_namespace(
    isolated_worker_cache, monkeypatch, tmp_path
):
    """The worker's in-memory hit must not skip the next job's disk export."""
    tests_dir = str(Path(__file__).resolve().parent)
    monkeypatch.syspath_prepend(tests_dir)
    import conftest as stub

    objects = []
    for version in (1, 2):
        helper_dir = tmp_path / f"helper-{version}"
        helper_dir.mkdir()
        (helper_dir / "helper.py").write_text(f"CODEGEN_VERSION = {version}\n")
        monkeypatch.setattr(cache_state, "EXTRA_SOURCE_DIRS", [helper_dir])
        jit_module._compute_source_fingerprint.cache_clear()
        sha, args, path = _request(stub.compile_stub, "same-key", isolated_worker_cache)
        err, compiled = ac._pool_worker(
            "conftest",
            "compile_stub",
            args,
            path,
            extra_source_dirs=(str(helper_dir),),
        )
        assert err is None
        assert set(compiled) == {sha}
        assert Path(path).read_bytes() == b"pool-shipped-stub"
        objects.append(path)

    assert objects[0] != objects[1]
    assert {str(p) for p in isolated_worker_cache.rglob("*.o")} == set(objects)


class _ManualExecutor:
    def __init__(self):
        self.submitted: list[tuple[Future, object, tuple]] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, fn, *args):
        future = Future()
        self.submitted.append((future, fn, args))
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))


def test_nested_pool_scope_and_suppression_restore_outer_pool(monkeypatch):
    """A nested prewarm/bench call must not discard the enclosing pool."""
    executor = _ManualExecutor()
    monkeypatch.setattr(ac, "_active_pool", None)
    monkeypatch.setattr(ac, "get_shared_executor", lambda: executor)
    with ac.pool_scope() as outer:
        assert ac.get_active_pool() is outer
        with ac.pool_scope() as inner:
            assert inner is outer
            with ac.suppress_pool():
                assert ac.get_active_pool() is None
            assert ac.get_active_pool() is outer
        assert ac.get_active_pool() is outer
    assert ac.get_active_pool() is None
    assert executor.shutdown_calls == []
    with pytest.raises(RuntimeError, match="compile caller failed"):
        with ac.pool_scope():
            raise RuntimeError("compile caller failed")
    assert ac.get_active_pool() is None


# ---------------------------------------------------------------------------
# End-to-end: defer-and-retry through a real pytest session
# ---------------------------------------------------------------------------

_INNER_TEST_SRC = textwrap.dedent(
    """
    import torch
    from cute_gemm.rmsnorm import rmsnorm_fwd

    def _run(N):
        x = torch.randn(32, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.float32)
        out, *_ = rmsnorm_fwd(x, w)
        ref = torch.nn.functional.rms_norm(x.float(), (N,), weight=w).to(x.dtype)
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=1e-2)

    def test_a():
        _run({N_A})

    def test_b():
        _run({N_B})
    """
)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_defer_and_retry_cold_session(tmp_path):
    """Cold keys + --async-compile: tests defer, retry, and verify numerics.

    Two test *modules* are used so the deferred retry crosses a module
    boundary — the regression that produced pytest's "previous item was not
    torn down properly" SetupState assert before the misprediction guard in
    ``_run_protocol`` (fixture teardown is scoped to a *predicted* next item
    and deferral breaks the prediction chain).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    # Unusual N values so the keys are cold in the throwaway cache dir.
    (suite / "test_defer_mod_a.py").write_text(_INNER_TEST_SRC.format(N_A=704, N_B=1216))
    (suite / "test_defer_mod_b.py").write_text(_INNER_TEST_SRC.format(N_A=1728, N_B=2240))
    (suite / "conftest.py").write_text('pytest_plugins = ["cute_gemm.testing.pytest_plugin"]\n')

    env = dict(os.environ)
    env["DLKERNEL_ARTIFACT_CACHE_DIR"] = str(tmp_path / "cache")

    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)  # pin tree under test
    env.pop("PYTEST_XDIST_WORKER", None)  # inner session is single-proc
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "4 passed" in out, out
    # The session must actually have exercised the pool + deferral path.
    assert "keys submitted" in out, out
    n_submitted = int(out.split("async-compile: ")[1].split(" keys")[0])
    assert n_submitted >= 4, out  # one fwd key per distinct N


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_warm_session_with_flag_submits_nothing(tmp_path):
    """Warm cache + --async-compile must be a no-op (no submits, no defers).

    Guards the lazy-pool property: enabling the flag unconditionally (e.g.
    in an alias or CI default) must cost nothing when the cache is warm.
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_warm.py").write_text(_INNER_TEST_SRC.format(N_A=704, N_B=1216))
    (suite / "conftest.py").write_text('pytest_plugins = ["cute_gemm.testing.pytest_plugin"]\n')

    env = dict(os.environ)
    env["DLKERNEL_ARTIFACT_CACHE_DIR"] = str(tmp_path / "cache")

    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)  # pin tree under test
    env.pop("PYTEST_XDIST_WORKER", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(suite),
        "-q",
        "-p",
        "no:cacheprovider",
        "--async-compile=4",
    ]
    kw = dict(capture_output=True, text=True, env=env, timeout=600, cwd=str(suite))
    first = subprocess.run(cmd, **kw)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(cmd, **kw)
    out = second.stdout + second.stderr
    assert second.returncode == 0, out
    assert "0 keys submitted" in out, out
    assert "0 test deferrals" in out, out


_WEDGED_POOL_CONFTEST_SRC = textwrap.dedent(
    """
    pytest_plugins = ["cute_gemm.testing.pytest_plugin"]


    def pytest_configure(config):
        # Simulate a wedged pool: submissions succeed but poll never resolves
        # — exactly what a hung pool worker (still holding the per-key flock)
        # looks like to every waiter. Shrink the drain wedge deadline so the
        # escape path runs quickly.
        from cute_gemm.cache import async_compile as ac
        from cute_gemm.testing import pytest_plugin as tp

        tp._XdistWorkerDefer._WEDGE_TIMEOUT_S = 2.0
        ac.CompilePool.poll = lambda self, sha: ("pending", None)
    """
)


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_xdist_drain_escapes_wedged_pool(tmp_path):
    """A sha that polls "pending" forever must not deadlock the final drain.

    Regression (Aug 2026, run 31081200282): pool compiles froze mid-job and
    every xdist worker's end-of-session drain skipped its still-pending items
    forever — ``_drain_ready`` never called ``_attempt`` on a pending sha, so
    the wedge escape's attempts bump never took effect and the CI leg died at
    ``timeout-minutes: 30``. With the fix, the wedge deadline forces the
    remaining items through in-process; without it, this inner session hangs
    (the outer ``timeout=`` converts that into a failure).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    # Unusual N values so the keys are cold in the throwaway cache dir.
    (suite / "test_wedged.py").write_text(_INNER_TEST_SRC.format(N_A=2752, N_B=3264))
    (suite / "conftest.py").write_text(_WEDGED_POOL_CONFTEST_SRC)

    env = dict(os.environ)
    env["DLKERNEL_ARTIFACT_CACHE_DIR"] = str(tmp_path / "cache")

    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)  # pin tree under test
    env.pop("PYTEST_XDIST_WORKER", None)  # the inner master must not be a worker
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=4",
            "-n",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "2 passed" in out, out


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="end-to-end defer needs a GPU"
)
def test_oom_retry_compile_pending_defers_not_warns(tmp_path):
    """OOM-retry hookwrapper + CompilePending on the retry must defer cleanly.

    Regression: tests/conftest.py's OOM-retry hook re-invokes
    ``item.runtest()`` inside an old-style hookwrapper teardown. When the
    retry hit a cold jit_cache miss (--async-compile), the resulting
    ``CompilePending`` escaped the teardown: pluggy emitted
    ``PluggyTeardownRaisedWarning``, the plugin's defer machinery (which had
    already run for this phase) never saw it, and the test reported as a
    CompilePending *failure* instead of deferring (or, worse, passing from a
    half-run attempt). Full analysis: the ``pytest_runtest_call`` docstring
    in tests/conftest.py.

    Uses the real repo conftest so the actual hook code is exercised. The
    inner test scripts the exact sequence: attempt 1 raises a synthetic OOM,
    the in-teardown retry raises CompilePending, and the defer-loop retry
    passes (the sha is unknown to the pool, so it polls "new" and re-runs
    immediately).
    """
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copy(Path(__file__).parent / "conftest.py", suite / "conftest.py")
    (suite / "test_oom_retry.py").write_text(
        textwrap.dedent(
            """
            import torch
            from cute_gemm.cache.async_compile import CompilePending

            CALLS = {"n": 0}

            def test_oom_then_cold_compile():
                CALLS["n"] += 1
                if CALLS["n"] == 1:
                    raise torch.OutOfMemoryError("synthetic CUDA out of memory")
                if CALLS["n"] == 2:
                    # This is the conftest OOM-retry's item.runtest() call:
                    # simulate the retry hitting a cold kernel compile.
                    raise CompilePending("a" * 64, "fake._compile_rmsnorm_bwd")
                assert CALLS["n"] == 3  # defer-loop retry succeeds
            """
        )
    )

    env = dict(os.environ)
    env["DLKERNEL_ARTIFACT_CACHE_DIR"] = str(tmp_path / "cache")

    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)  # pin tree under test
    env.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--async-compile=2",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        cwd=str(suite),
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"inner pytest failed:\n{out}"
    assert "1 passed" in out, out
    assert "PluggyTeardownRaisedWarning" not in out, out
    assert "1 test deferrals" in out, out


def test_gpu_blind_device_attr_shim():
    """GPU-blind workers must answer the DSL's one trace-time driver query
    (max smem per SM, needed when a launch sets min_blocks_per_mp > 1) from
    the static arch table — cuDeviceGetAttribute raises NOT_INITIALIZED with
    CUDA_VISIBLE_DEVICES="" and would fail every occupancy-bound kernel key
    (e.g. the W4 small-N decode configs) over to in-process compiles."""
    code = textwrap.dedent(
        """
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CUTE_DSL_ARCH"] = "sm_90a"
        from cute_gemm.cache.async_compile import _install_gpu_blind_device_attrs

        _install_gpu_blind_device_attrs()
        from cutlass.base_dsl.runtime import cuda as cuda_helpers

        dev_attrs = cuda_helpers.cuda.CUdevice_attribute
        attr = dev_attrs.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        # sm_90 SM total: the DSL's per-CTA capacity + the 1 KiB reserved slice
        assert cuda_helpers.get_device_attribute(attr) == 233472
        print("shim-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "shim-ok" in result.stdout, result.stderr


def test_pool_arch_pin_survives_early_dsl_construction():
    """The worker's ptxas target must land on a DSL singleton that was built
    BEFORE the pool's env pinning ran.

    Regression (cutlass-dsl 4.6.2 CI image): importing
    ``cute_gemm.cache._pool_preload`` executes the parent packages first, and
    ``cute_gemm/__init__`` pulls in cutlass — constructing the DSL env manager
    before ``CUTE_DSL_ARCH`` is set. 4.6.2 snapshots the env var at
    construction (4.6.0 read it lazily), so env pinning alone left the
    singleton unlatched; a GPU-blind worker's first ``.arch`` read then fell
    back to ``detect_gpu_arch()``'s sm_100a default, and every pool ``.o``
    on the B300 runner (sm_103a) failed to load with
    cudaErrorNoKernelImageForDevice — which the 4.6.2 runtime escalated into
    a permanent spinlock hang on the process's next launch. The explicit
    ``_pin_dsl_arch`` re-latch in ``_pool_initializer`` / ``_pool_preload``
    is what this test protects.
    """
    code = textwrap.dedent(
        """
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("CUTE_DSL_ARCH", None)
        # Simulate the parent-package import chain: cutlass (and the DSL
        # env-manager singleton) come up with no CUTE_DSL_ARCH in sight.
        import cute_gemm.cache  # noqa: F401

        # The pool's env pinning + re-latch, as _pool_initializer runs them.
        from cute_gemm.cache.async_compile import _pool_initializer

        _pool_initializer("90", "sm_90a")

        from cutlass.cutlass_dsl import CuTeDSL

        arch = CuTeDSL._get_dsl().envar.arch
        assert arch == "sm_90a", f"worker would compile for {arch}, not sm_90a"
        print("arch-pin-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "arch-pin-ok" in result.stdout, result.stderr


def test_detect_arch_env_keeps_physical_dispatch_for_cross_target(monkeypatch):
    """An explicit CuTe target must not replace physical dispatch metadata."""
    import sys
    import types

    from cute_gemm.cache.async_compile import _detect_arch_env

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (9, 0),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("DLKERNEL_ARCH", raising=False)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")

    assert _detect_arch_env() == ("90", "sm_100a")
