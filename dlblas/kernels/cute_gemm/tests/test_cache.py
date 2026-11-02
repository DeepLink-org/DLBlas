# Copyright (c) 2026, Tri Dao.
"""Unit tests for ``cute_gemm.cache``.

Most of these are deliberately *small and synchronous*: they exercise the
package's mutable-state plumbing without launching real CuTe kernels, so they
catch import-order, mutation-visibility, and exception-propagation regressions
fast.  The shared-directory contention section near the end uses threads and
stubbed subprocesses, but keeps the same no-real-CuTe-kernel rule.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest


# Note: ``cute_gemm.cache`` is intentionally re-imported inside each test below;
# the inside-function import makes the dependency explicit at the test
# boundary.


# ---------------------------------------------------------------------------
# Package layout / public-API surface
# ---------------------------------------------------------------------------


def test_public_api_symbols_resolve():
    """Every name advertised in ``cute_gemm.cache.__all__`` must actually exist
    on the package and be importable as ``from cute_gemm.cache import X``.

    This is the regression test for the brittle import order in
    ``__init__.py``: if a maintainer reorders flag defs vs. submodule
    imports, the package will fail to initialize and this test fires.
    """
    import cute_gemm.cache

    for name in cute_gemm.cache.__all__:
        assert hasattr(cute_gemm.cache, name), (
            f"cute_gemm.cache advertises {name!r} in __all__ but it's missing"
        )


def test_static_config_flags_live_on_package_init():
    """Static-config flags must be direct attributes of the ``cute_gemm.cache``
    package object so that callers can set them once (before importing
    submodules that read them).

    """
    import cute_gemm.cache

    for name in ("CACHE_ENABLED", "CACHE_DIR", "EXTRA_SOURCE_DIRS"):
        assert name in vars(cute_gemm.cache), (
            f"static-config flag {name!r} should be a direct attribute of cute_gemm.cache, not a re-export"
        )


def test_jit_module_sees_live_flags():
    """``cute_gemm.cache.jit`` reads flags via ``_state`` (the partially-imported
    package). A regression that breaks this would surface as the disk cache
    silently ignoring ``CACHE_ENABLED=0``.
    """
    import cute_gemm.cache
    import cute_gemm.cache.jit as jit_module

    # `_state` should be the package object itself.
    assert jit_module._state is cute_gemm.cache


def test_jit_cache_identity_separates_binary_targets_without_device_ordinal():
    """The persistent object namespace must identify executable artifacts.

    An autotune result can be correct while the compiled ``.o`` belongs to a
    different CuTe target or dispatch architecture.  Conversely, two cards
    with the same architecture must share artifacts even when
    ``CUDA_VISIBLE_DEVICES`` remaps their ordinals.  This is deliberately a
    pure-host test: it must not initialize CUDA or inspect tensor values.
    """
    from cute_gemm.cache import jit as jit_module

    base = jit_module._cache_environment_identity(
        cute_dsl_arch="sm_90a",
        dlkernel_arch="90",
        physical_arch="sm_90a",
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    target_changed = jit_module._cache_environment_identity(
        cute_dsl_arch="sm_100a",
        dlkernel_arch="90",
        physical_arch="sm_90a",
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    dispatch_changed = jit_module._cache_environment_identity(
        cute_dsl_arch="sm_90a",
        dlkernel_arch="100",
        physical_arch="sm_90a",
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    abi_changed = jit_module._cache_environment_identity(
        cute_dsl_arch="sm_90a",
        dlkernel_arch="90",
        physical_arch="sm_90a",
        cuda_version="13.0",
        torch_version="2.8.0",
    )

    assert base != target_changed
    assert base != dispatch_changed
    assert base != abi_changed

    # The helper has no GPU ordinal argument by design.  A CUDA_VISIBLE_DEVICES
    # remap therefore cannot split artifacts that are executable on the same
    # architecture.
    assert base == jit_module._cache_environment_identity(
        cute_dsl_arch="sm_90a",
        dlkernel_arch="90",
        physical_arch="sm_90a",
        cuda_version="12.8",
        torch_version="2.8.0",
    )


def test_jit_source_fingerprint_includes_cache_environment_identity(monkeypatch):
    """Changing the effective target invalidates the source-fingerprint path."""
    from cute_gemm.cache import jit as jit_module

    monkeypatch.setattr(jit_module, "_get_physical_arch", lambda: "sm_90a")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    monkeypatch.setenv("DLKERNEL_ARCH", "90")
    jit_module._compute_source_fingerprint.cache_clear()
    first = jit_module._compute_source_fingerprint()

    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    jit_module._compute_source_fingerprint.cache_clear()
    second = jit_module._compute_source_fingerprint()

    assert first != second
    # Leave a clean cache behind: without this, the fingerprint computed
    # under the monkeypatched env poisons every later in-process compile
    # (and diverges from async-pool workers, which compute it fresh).
    jit_module._compute_source_fingerprint.cache_clear()


def test_jit_cache_identity_matches_gpu_blind_worker_defaults():
    """A worker's derived ``90`` dispatch must match the parent's ``sm_90a``."""
    from cute_gemm.cache import jit as jit_module

    parent = jit_module._cache_environment_identity(
        physical_arch="sm_90a",
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    worker = jit_module._cache_environment_identity(
        cute_dsl_arch="sm_90a",
        dlkernel_arch="90",
        physical_arch=None,
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    assert parent == worker


def test_jit_cache_identity_matches_cpu_cross_compile_defaults():
    """A dispatch-only CPU compile must derive the same target as its worker."""
    from cute_gemm.cache import jit as jit_module

    parent = jit_module._cache_environment_identity(
        dlkernel_arch="90",
        physical_arch=None,
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    worker = jit_module._cache_environment_identity(
        cute_dsl_arch="sm_90a",
        dlkernel_arch="90",
        physical_arch=None,
        cuda_version="12.8",
        torch_version="2.8.0",
    )
    assert parent == worker


def test_hadamard_codegen_flag_changes_compile_request(monkeypatch):
    """A schedule-changing env flag must reach both the key and kernel plan."""
    from cutlass import BFloat16

    from cute_gemm.cache import CompileCaptured, capture_compiles
    from cute_gemm.transform.hadamard import HadamardTransform, _get_compiled_hadamard

    requests = {}
    for value in ("0", "1"):
        monkeypatch.setenv("DLKERNEL_HADAMARD_TAIL_DIRECT_STORE", value)
        with capture_compiles() as captured:
            with pytest.raises(CompileCaptured):
                _get_compiled_hadamard(BFloat16, 512, False)
        (spec,) = captured.tasks()
        requests[value] = spec
        assert spec.args[-1] is (value == "1")

    assert requests["0"].sha != requests["1"].sha
    assert requests["0"].o_path != requests["1"].o_path
    disabled = HadamardTransform(BFloat16, 512, tail_direct_store_enabled=False)
    enabled = HadamardTransform(BFloat16, 512, tail_direct_store_enabled=True)
    assert not disabled.plan.use_tail_direct_store
    assert enabled.plan.use_tail_direct_store


# ---------------------------------------------------------------------------
# Regression: lock-before-compile prevents the cold-cache convoy
# ---------------------------------------------------------------------------


def test_jit_cache_lock_serializes_redundant_compiles(tmp_path):
    """N concurrent processes hitting the same cold key compile exactly once.

    The pre-fix behavior of ``jit_cache.wrapper`` was: check disk under a
    shared lock; on miss, release the shared lock, run ``fn(*args, **kwargs)``
    *unlocked*, then take an exclusive lock to export. Under N-way racing
    that meant **N processes compiled the same key in parallel** — wall
    time was bounded by one compile, but compile-CPU pressure scaled with
    N, starving other compiles when many keys were cold at once.

    Post-fix: the compile body runs under the exclusive lock, with a
    re-check inside the lock. N-1 of the racing processes wait on the
    lock, then see the ``.o`` and load it.

    We verify by spawning N subprocesses that all call the same
    ``@jit_cache``-decorated stub. Each invocation of the compile body
    writes a marker file; we count markers. Without the fix: ~N markers.
    With the fix: 1.
    """
    import subprocess
    import sys
    import textwrap

    cache_dir = tmp_path / "cache"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    # Inline child script. We monkey-patch ``cute.runtime.load_module`` before
    # importing ``cute_gemm.cache`` so the load path returns a harmless stub
    # (the ``.o`` written by export is a placeholder, not a real CuTe object).
    # The actual thing under test is purely the lock + recheck logic in
    # ``jit_cache.wrapper``; no GPU or real CuTe involvement is needed.
    # Barrier file: subprocess imports are slow (~1-3 s for cute.compile),
    # so we can't rely on launching them to make them race. Each child does
    # all its imports / monkeypatching first, then spins on a barrier file
    # until the parent writes it. That guarantees all N children enter
    # ``_compile_stub(...)`` within a tight window.
    barrier = tmp_path / "GO"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        textwrap.dedent(
            f"""
            import os, sys, time
            from pathlib import Path

            # Stub cute.runtime.load_module BEFORE importing cute_gemm.cache so
            # downstream readers (post-compile load path) don't choke on the
            # placeholder .o we'll write.
            import cutlass.cute as _cute
            class _StubMod:
                def __getitem__(self, name):
                    return lambda *a, **k: None
            _cute.runtime.load_module = lambda path, enable_tvm_ffi=True: _StubMod()

            import cute_gemm.cache

            MARKER_DIR = Path({str(marker_dir)!r})
            BARRIER = Path({str(barrier)!r})

            @cute_gemm.cache.jit_cache
            def _compile_stub(key):
                # Atomically record that THIS process ran the compile body.
                # Filename includes pid + ns clock so concurrent writes never
                # collide on the same marker name.
                marker = MARKER_DIR / f"compile_{{os.getpid()}}_{{time.time_ns()}}.marker"
                marker.touch()
                # Hold long enough that the *next* worker, if it slipped
                # through the unlocked-compile bug, would also be inside
                # this body simultaneously. We test with 1.5 s headroom;
                # post-fix the exclusive lock serializes us, so subsequent
                # workers wait, see the .o, and load without re-entering.
                time.sleep(1.5)

                class _Compiled:
                    def export_to_c(self, object_file_path, function_name):
                        Path(object_file_path).write_bytes(b"placeholder-o")

                    def __call__(self, *a, **k):
                        pass

                return _Compiled()

            # Spin on the barrier so all N children enter the wrapper
            # within a few ms of each other. Bound the wait so a stuck
            # parent doesn't hang the test.
            deadline = time.time() + 60
            while not BARRIER.exists():
                if time.time() > deadline:
                    sys.exit("barrier timeout")
                time.sleep(0.01)

            _compile_stub("shared-cold-key")
            """
        )
    )

    env = {
        **os.environ,
        "DLKERNEL_ARTIFACT_CACHE_DIR": str(cache_dir),
        "DLKERNEL_CACHE_ENABLED": "1",
        # Children run `python <script>`: sys.path[0] is the script's dir, not
        # the repo, so without this pin they import whichever cute_gemm the
        # editable install points at — a different tree with different cache
        # semantics — and silently test the wrong code.
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }

    N = 8
    procs = [
        subprocess.Popen(
            [sys.executable, str(child_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(N)
    ]
    # Give all N children time to finish their imports and reach the barrier
    # spin-loop. A few seconds is plenty even for the slow cute imports;
    # children that haven't reached the barrier by the time we flip it will
    # just enter slightly later, which still tests the race we care about.
    import time as _time

    _time.sleep(5.0)
    barrier.touch()

    # Wait for all; collect output for diagnostics if anything failed.
    outputs = [(p.wait(timeout=120), p.stdout.read(), p.stderr.read()) for p in procs]
    for rc, out, err in outputs:
        assert rc == 0, (
            f"child subprocess exited non-zero ({rc})\n"
            f"stdout:\n{out.decode(errors='replace')}\n"
            f"stderr:\n{err.decode(errors='replace')}"
        )

    markers = list(marker_dir.glob("compile_*.marker"))
    assert len(markers) == 1, (
        f"expected exactly 1 compile invocation across {N} racing processes "
        f"(lock-before-compile should serialize), saw {len(markers)}: {markers!r}"
    )


# ---------------------------------------------------------------------------
# R3b: object-cache dir decoupling (DLKERNEL_ARTIFACT_CACHE_DIR)
# ---------------------------------------------------------------------------


def test_object_cache_dir_env_matrix(tmp_path):
    """``DLKERNEL_ARTIFACT_CACHE_DIR`` steers ONLY the ``.o`` artifact cache.

    Pure two-knob topology (no cross-fallback): the artifact cache follows
    its own variable and ignores ``DLKERNEL_CACHE_DIR`` (which belongs to the
    autotuner's winner store; that side is covered in test_autotuner). The
    env is parsed at package import, so each combination runs in a fresh
    subprocess (a reload would violate the documented partial-import ordering
    of ``cute_gemm.cache``). The matrix pins the contract:

    * only the artifact var set   → artifact cache under it;
    * both set (different trees)  → artifact cache follows the artifact var;
    * only ``DLKERNEL_CACHE_DIR`` → artifact cache does NOT move (checked
      separately below: stays on the per-user default, outside tmp_path).
    """
    import subprocess
    import sys

    artifact_root, only_artifact_root = (tmp_path / n for n in ("a", "b"))
    tune_only_root = tmp_path / "tune-only"
    cases = [
        ({"DLKERNEL_ARTIFACT_CACHE_DIR": str(only_artifact_root)}, only_artifact_root),
        # The two vars point at DIFFERENT trees: the artifact cache must
        # follow its own var, not the winner-cache var.
        (
            {
                "DLKERNEL_CACHE_DIR": str(tune_only_root),
                "DLKERNEL_ARTIFACT_CACHE_DIR": str(artifact_root),
            },
            artifact_root,
        ),
    ]
    child = "import cute_gemm.cache as c\nprint(c.get_cache_path())\n"
    for env_overrides, expected_root in cases:
        env = {
            **{
                k: v
                for k, v in os.environ.items()
                if k not in ("DLKERNEL_CACHE_DIR", "DLKERNEL_ARTIFACT_CACHE_DIR")
            },
            **env_overrides,
            # Pin the tree under test: `python -c` puts cwd on sys.path, but
            # this must not depend on where pytest was invoked from.
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        }
        out = subprocess.run(
            [sys.executable, "-c", child],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert out.returncode == 0, out.stderr
        reported = out.stdout.strip().splitlines()[-1]
        assert Path(reported).is_relative_to(expected_root), (
            f"object cache {reported!r} not under expected root {expected_root} "
            f"for env {env_overrides}"
        )

    # The winner-cache var alone must NOT steer artifacts anywhere under the
    # test tree: artifacts stay on the per-user default root.
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if k not in ("DLKERNEL_CACHE_DIR", "DLKERNEL_ARTIFACT_CACHE_DIR")
        },
        "DLKERNEL_CACHE_DIR": str(tune_only_root),
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
    }
    out = subprocess.run(
        [sys.executable, "-c", child], env=env, capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr
    reported = out.stdout.strip().splitlines()[-1]
    assert not Path(reported).is_relative_to(tmp_path), (
        f"artifact cache must ignore DLKERNEL_CACHE_DIR, moved to {reported!r}"
    )


# ---------------------------------------------------------------------------
# R3c: capture mode + precompile (framework prewarming API)
# ---------------------------------------------------------------------------


class _PlaceholderCompiled:
    """Stand-in for a cute.compile product: exports a placeholder .o."""

    def export_to_c(self, object_file_path, function_name):
        from pathlib import Path

        Path(object_file_path).write_bytes(b"placeholder-o")

    def __call__(self, *a, **k):
        pass


def test_capture_compiles_records_and_aborts(tmp_path, monkeypatch):
    """Capture mode records jit_cache boundaries and aborts the call.

    The abort must be a BaseException (an ``except Exception`` around the
    driving call cannot swallow it), duplicate keys dedupe by sha, and the
    wrapper behaves normally once capture exits.
    """
    import cute_gemm.cache
    from cute_gemm.cache import jit_cache
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))

    @jit_cache
    def _stub_kernel(scale):
        return _PlaceholderCompiled()

    with capture_compiles() as captured:
        with pytest.raises(CompileCaptured):
            _stub_kernel(3)
        # Same key again inside one capture context: recorded once (sha-deduped).
        with pytest.raises(CompileCaptured):
            _stub_kernel(3)
        with pytest.raises(CompileCaptured):
            _stub_kernel(4)
    tasks = captured.tasks()
    assert len(tasks) == 2, [t[0][:12] for t in tasks]
    assert tasks[0][0] != tasks[1][0]

    # Capture is BaseException-derived: unreachable via except Exception.
    assert issubclass(CompileCaptured, BaseException)
    assert not issubclass(CompileCaptured, Exception)

    # Outside capture the wrapper compiles (or loads) as usual.
    result = _stub_kernel(3)
    assert callable(result)
    assert _stub_kernel.cache_info().currsize == 1


def _patch_load_module(monkeypatch, payload_tag: bool = False):
    """Make warm ``.o`` loads succeed on placeholder bytes (no real CuTe load).

    With ``payload_tag`` the loaded "function" exposes the artifact's bytes as
    ``.tag`` (decoded), so tests can prove *which* module's artifact a call
    actually loaded.
    """
    import cutlass.cute as cute

    class _Loaded:
        def __init__(self, payload):
            self.tag = payload.decode()

    class _StubMod:
        def __init__(self, path):
            from pathlib import Path

            self.payload = Path(path).read_bytes()

        def __getitem__(self, name):
            if payload_tag:
                return _Loaded(self.payload)
            return lambda *a, **k: None

    monkeypatch.setattr(
        cute.runtime, "load_module", lambda path, enable_tvm_ffi=True: _StubMod(path)
    )


def test_precompile_fills_object_cache_without_duplicates(tmp_path, monkeypatch):
    """``precompile`` discovers via capture, dedupes, and compiles every key.

    A ``<locals>`` stub cannot be shipped to a pool worker, so the compile
    runs in-process — exercising the same jit_cache slow path (export under
    lock) the pool workers run. Second invocation re-loads from disk
    (materialize evicts in-memory entries) and reports pure hits.
    """
    import cute_gemm.cache
    from cute_gemm.cache import jit_cache, precompile

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))
    _patch_load_module(monkeypatch)

    @jit_cache
    def _stub_kernel(scale):
        return _PlaceholderCompiled()

    report = precompile(
        [
            lambda: _stub_kernel(7),
            lambda: _stub_kernel(7),  # duplicate specialization
            lambda: _stub_kernel(9),
        ]
    )
    assert report.n_calls == 3
    assert report.n_unique == 2
    assert report.n_compiled == 2
    assert report.n_hits == 0
    objects = list((tmp_path / "objects").rglob("*.o"))
    assert len(objects) == 2, objects

    misses_before = _stub_kernel.cache_info().misses
    report2 = precompile([lambda: _stub_kernel(7), lambda: _stub_kernel(9)])
    assert report2.n_unique == 2
    assert report2.n_compiled == 0
    assert report2.n_hits == 2
    assert _stub_kernel.cache_info().misses == misses_before  # no recompile


def test_precompile_exercises_async_pool(tmp_path, monkeypatch):
    """precompile ships module-level stubs through the REAL pool path.

    Covers the machinery the ``<locals>`` tests cannot: submit →
    CompilePending → rotation → poll(done) → disk load. The artifact's bytes
    prove a worker compiled it (the parent never runs the compile body).
    """
    import sys

    import cute_gemm.cache
    from cute_gemm.cache import pool_scope, precompile

    # `tests` resolves to an unrelated site-packages package on this host;
    # import the stub top-level so pool workers can resolve it by name too.
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import conftest as stub

    stub.compile_stub.cache_clear()  # module-level stub: memory cache survives tests

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("DLKERNEL_ARCH", "90")  # deterministic GPU-blind workers
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    _patch_load_module(monkeypatch)
    # Compute the fingerprint under THIS test's env so the parent's artifact
    # path matches what fresh worker processes derive.
    import cute_gemm.cache.jit as jit_module

    jit_module._compute_source_fingerprint.cache_clear()
    try:
        report = precompile([lambda: stub.compile_stub("pool")])
        assert report.n_unique == 1
        assert report.n_compiled == 1
        assert report.n_hits == 0
        objects = list((tmp_path / "objects").rglob("*.o"))
        assert len(objects) == 1, objects
        assert objects[0].read_bytes() == b"pool-shipped-stub"

        report2 = precompile([lambda: stub.compile_stub("pool")])
        assert report2.n_compiled == 0
        assert report2.n_hits == 1

        # A caller may keep the pool active across precompile calls. The
        # second call must not claim compiles retained in that pool's ledger.
        with pool_scope():
            first = precompile([lambda: stub.compile_stub("pool-active")])
            second = precompile([lambda: stub.compile_stub("pool-active")])
        assert first.n_compiled == 1
        assert second.n_compiled == 0
        assert second.n_hits == 1
    finally:
        # Leave a clean cache behind: the fingerprint computed under this
        # test's monkeypatched arch env would otherwise stay cached and
        # poison every later in-process compile — on GPU-visible CI the
        # real arch diverges from it (and from fresh pool workers).
        jit_module._compute_source_fingerprint.cache_clear()


def test_precompile_restores_missing_artifacts(tmp_path, monkeypatch):
    """A warm process must not satisfy materialization from memory.

    Regression (external review 2026-09-23): with the artifact deleted,
    ``precompile`` used to report ``n_compiled=0, n_hits=1`` and leave zero
    ``.o`` files — a "successful" prewarm with nothing on disk. Materialize
    now evicts in-memory entries (every call exercises the disk path) and
    treats a still-missing artifact as an error.
    """
    import cute_gemm.cache
    from cute_gemm.cache import jit_cache, precompile

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))
    _patch_load_module(monkeypatch)

    @jit_cache
    def _stub_kernel(scale):
        return _PlaceholderCompiled()

    precompile([lambda: _stub_kernel(5)])
    for o in (tmp_path / "objects").rglob("*.o"):
        o.unlink()

    report = precompile([lambda: _stub_kernel(5)])
    assert report.n_compiled == 1
    assert report.n_hits == 0
    assert len(list((tmp_path / "objects").rglob("*.o"))) == 1


def test_precompile_raises_when_export_persistently_fails(tmp_path, monkeypatch):
    """A persistently failing export must raise, never report a fake success.

    This is the fuse ``run_varlen_moe_validation.sh``'s PREWARM gate depends
    on: if the existence post-condition were weakened to a warning, the
    prewarm would "succeed" with zero artifacts and 1294 candidates would
    cold-compile inside the GPU jobs while every test stays green. The
    export warning still fires first (warn-then-raise, not warn-and-return).
    """
    import cute_gemm.cache
    from cute_gemm.cache import jit_cache, precompile

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))
    _patch_load_module(monkeypatch)

    class _BrokenExport:
        def export_to_c(self, object_file_path, function_name):
            raise OSError("simulated persistent export failure")

        def __call__(self, *a, **k):
            pass

    @jit_cache
    def _stub_kernel(scale):
        return _BrokenExport()

    with pytest.warns(RuntimeWarning, match="export failed"):
        with pytest.raises(RuntimeError, match="materialize incomplete"):
            precompile([lambda: _stub_kernel(3)])
    assert not list((tmp_path / "objects").rglob("*.o"))


def test_precompile_rejects_disabled_cache_before_running_calls(monkeypatch):
    """Disabled cache cannot capture boundaries; calls may have GPU effects."""
    import cute_gemm.cache
    from cute_gemm.cache import precompile

    monkeypatch.setattr(cute_gemm.cache, "CACHE_ENABLED", False)
    calls_run = []

    with pytest.raises(RuntimeError, match="DLKERNEL_CACHE_ENABLED=1"):
        precompile([lambda: calls_run.append("unexpected call")])
    assert calls_run == []


def test_precompile_rejects_call_without_compile_boundary():
    """An incomplete call inventory must not report a successful prewarm."""
    from cute_gemm.cache import precompile

    calls_run = []
    with pytest.raises(RuntimeError, match="did not reach a jit_cache boundary"):
        precompile([lambda: calls_run.append("ran")])
    assert calls_run == ["ran"]


def test_materialize_rejects_stale_captured_cache_root(tmp_path, monkeypatch):
    """A captured recipe must not report success against a different root."""
    import cute_gemm.cache
    from cute_gemm.cache import jit_cache
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles
    from cute_gemm.cache.precompile import materialize

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "before"))
    compiled = []

    @jit_cache
    def _stub_kernel(key):
        compiled.append(key)
        return _PlaceholderCompiled()

    with capture_compiles() as captured:
        with pytest.raises(CompileCaptured):
            _stub_kernel("k")
    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "after"))
    with pytest.raises(ValueError, match="stale CompileSpec"):
        materialize(captured.tasks())
    assert compiled == []
    assert not list(tmp_path.rglob("*.o"))


def test_materialize_rejects_relative_cache_root_after_chdir(tmp_path, monkeypatch):
    """A relative root refers to a new tree after chdir, even with the same text."""
    import cute_gemm.cache
    from cute_gemm.cache import jit_cache
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles
    from cute_gemm.cache.precompile import materialize

    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    monkeypatch.chdir(before)
    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", "objects")
    compiled = []

    @jit_cache
    def _stub_kernel(key):
        compiled.append(key)
        return _PlaceholderCompiled()

    with capture_compiles() as captured:
        with pytest.raises(CompileCaptured):
            _stub_kernel("k")
    monkeypatch.chdir(after)
    with pytest.raises(ValueError, match="stale CompileSpec"):
        materialize(captured.tasks())
    assert compiled == []
    assert not list(tmp_path.rglob("*.o"))


def test_cache_key_separates_same_named_functions_across_modules(tmp_path, monkeypatch):
    """Two modules defining same-named compile functions must not share one ``.o``.

    Regression (external review 2026-09-23): the disk key used only
    ``fn.__qualname__`` + args, so the second module silently loaded the
    first module's artifact (reproduced: one ``.o``, wrong payload). The key
    now includes ``fn.__module__``.
    """
    import sys
    import types

    import cute_gemm.cache

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import conftest as stub

    stub.compile_stub.cache_clear()  # module-level stub: memory cache survives tests

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))
    _patch_load_module(monkeypatch, payload_tag=True)

    twin_src = """
from pathlib import Path
from cute_gemm.cache import jit_cache

@jit_cache
def compile_stub(key):
    class _TwinCompiled:
        tag = "twin-stub"  # same value fresh-compiled and payload-loaded
        def export_to_c(self, object_file_path, function_name):
            Path(object_file_path).write_bytes(b"twin-stub")
        def __call__(self, *a, **k):
            pass
    return _TwinCompiled()
"""
    twin = types.ModuleType("dlkernel_cache_test_twin")
    exec(twin_src, twin.__dict__)
    sys.modules["dlkernel_cache_test_twin"] = twin

    assert stub.compile_stub("k").key == "k"  # compiles, writes pool-shipped-stub
    loaded = twin.compile_stub("k")  # must NOT load the stub module's .o
    assert loaded.tag == "twin-stub", f"cross-module artifact confusion: twin loaded {loaded.tag!r}"
    payloads = sorted(p.read_bytes() for p in (tmp_path / "objects").rglob("*.o"))
    assert payloads == [b"pool-shipped-stub", b"twin-stub"]


def test_external_module_source_change_invalidates_key(tmp_path, monkeypatch):
    """An edit to an *unregistered* external module must invalidate the disk key.

    The external module lives outside ``EXTRA_SOURCE_DIRS``, so the
    package-level source fingerprint never sees it; only the per-request sha
    can carry the defining module's source-file digest. Keep the module name,
    size and mtime identical after the edit: none can accidentally distinguish
    v1 from v2. An old live wrapper must also keep its v1 identity.
    """
    import sys
    import types

    import cute_gemm.cache

    def _ext_src(tag: str) -> str:
        return f"""\
from pathlib import Path

from cute_gemm.cache import jit_cache


class _Compiled:
    tag = {tag!r}  # same value fresh-compiled and payload-loaded

    def export_to_c(self, object_file_path, function_name):
        Path(object_file_path).write_bytes({tag.encode()!r})

    def __call__(self, *args, **kwargs):
        pass


@jit_cache
def compile_stub(key: str):
    return _Compiled()
"""

    def _load(name):
        module = types.ModuleType(name)
        module.__file__ = str(ext)
        sys.modules[name] = module
        exec(compile(ext.read_text(), str(ext), "exec"), module.__dict__)
        return module

    ext = tmp_path / "extmod_v1.py"
    ext.write_text(_ext_src("v1"))
    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "objects"))
    _patch_load_module(monkeypatch, payload_tag=True)

    mod1 = _load("extmod_under_test")
    first = mod1.compile_stub("k")
    assert first.tag == "v1"

    old_stat = ext.stat()
    ext.write_text(_ext_src("v2"))
    assert ext.stat().st_size == old_stat.st_size
    os.utime(ext, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert ext.stat().st_mtime_ns == old_stat.st_mtime_ns

    # This old callable remains valid in a long-lived process. It must not
    # publish v1 code under the newly edited v2 source identity.
    assert mod1.compile_stub("after-edit").tag == "v1"

    sys.modules.pop("extmod_under_test", None)
    mod2 = _load("extmod_under_test")
    second = mod2.compile_stub("after-edit")
    assert second.tag == "v2", (
        f"edited external module must recompile, loaded stale tag {second.tag!r}"
    )

    objects = list((tmp_path / "objects").rglob("*.o"))
    assert len(objects) == 3, objects
    assert {o.read_bytes() for o in objects} == {b"v1", b"v2"}
    sys.modules.pop("extmod_under_test_v2", None)


# ---------------------------------------------------------------------------
# R3e: DeepGEMM-style no-clobber artifact publish (jit.py _publish_artifact)
# ---------------------------------------------------------------------------


def test_publish_artifact_no_clobber_race(tmp_path):
    """Concurrent publishers of one key: exactly one wins, losers reuse.

    The publish primitive's correctness must come from the filesystem
    (os.link atomicity), not from any lock: N threads racing distinct
    exports at one target must yield exactly one winner whose bytes are
    what every loser now sees — never a clobbered or torn file.
    """
    from concurrent.futures import ThreadPoolExecutor

    from cute_gemm.cache.jit import _publish_artifact

    target = tmp_path / "kernel.o"
    tmps = []
    for i in range(8):
        t = tmp_path / f"tmp-{i}.o"
        t.write_bytes(f"export-{i}".encode())
        tmps.append(t)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda t: _publish_artifact(t, target), tmps))

    assert sum(results) == 1, results  # exactly one publisher won
    winner_bytes = target.read_bytes()
    assert winner_bytes.startswith(b"export-")
    # Every loser's temp file was cleaned up; the target is a complete,
    # single-publisher artifact (never concatenated or partial).
    assert len(winner_bytes) == len("export-0")
    for t in tmps:
        assert not t.exists(), t

    # A later publisher of the same key loses and keeps the existing bytes.
    late = tmp_path / "tmp-late.o"
    late.write_bytes(b"export-late")
    assert _publish_artifact(late, target) is False
    assert target.read_bytes() == winner_bytes
    assert not late.exists()


# ---------------------------------------------------------------------------
# Shared-directory contention: corrupt repair, timeouts, crashes, pools
# ---------------------------------------------------------------------------


class _IntArtifact:
    """Integer stand-in for a cute.compile product: the ``.o`` payload is the
    decimal value, and calling the loaded module returns it.
    """

    def __init__(self, value: int):
        self.value = value

    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        Path(object_file_path).write_text(str(self.value))

    def __call__(self) -> int:
        return self.value


@pytest.fixture
def isolated_object_cache(tmp_path, monkeypatch):
    """Point the package object cache at an empty tmp dir, with stubbed loads.

    Keeps the tests below focused on shared-directory file behavior and stable
    if another test changes the process-wide fingerprint cache or GPU override.
    """
    import cutlass.cute as cute

    import cute_gemm.cache as cache
    from cute_gemm.cache import jit

    repo_root = Path(__file__).resolve().parent.parent
    assert Path(jit.__file__).resolve().is_relative_to(repo_root), jit.__file__
    root = tmp_path / "shared-objects"
    monkeypatch.setattr(cache, "CACHE_DIR", str(root))
    monkeypatch.setattr(cache, "CACHE_ENABLED", True)
    monkeypatch.setattr(jit, "_compute_source_fingerprint", lambda: "stress-namespace")

    class _Module:
        def __init__(self, path):
            self.path = Path(path)

        def __getitem__(self, name):
            assert name == "func"
            return _IntArtifact(int(self.path.read_text()))

    monkeypatch.setattr(cute.runtime, "load_module", lambda path, **_: _Module(path))
    return root


def test_corrupt_object_is_quarantined_and_recompiled(isolated_object_cache):
    """A truncated persistent object is deleted and recompiled, not served.

    Concurrency note: the jit_cache wrapper is single-threaded by contract
    (no lock on the in-memory dict; cute.compile is not thread-safe), so the
    corrupt-repair *behavior* is pinned here single-threadedly; the
    concurrent-repair corner is exercised by the shared-directory stress
    suite (S3), which drives whole processes instead of threads.
    """
    from cute_gemm.cache import jit as jit_module
    from cute_gemm.cache import jit_cache

    compile_count = 0

    @jit_cache
    def compile_value(value):
        nonlocal compile_count
        compile_count += 1
        return _IntArtifact(value)

    sha = jit_module._request_sha(compile_value.__wrapped__, (37,))
    object_path = isolated_object_cache / "stress-namespace" / f"{sha}.o"
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"truncated-object")

    with pytest.warns(RuntimeWarning, match="corrupt cached object"):
        assert compile_value(37)() == 37

    assert compile_count == 1
    assert object_path.read_text() == "37"


def test_failed_export_leaves_no_published_or_private_object(isolated_object_cache):
    """Disk or exporter errors must not publish a partial .o or leak a temp file."""
    from cute_gemm.cache import jit_cache

    fail_export = True
    compile_count = 0

    class _FlakyArtifact(_IntArtifact):
        def export_to_c(self, object_file_path, function_name):
            if fail_export:
                Path(object_file_path).write_bytes(b"partial")
                raise OSError("simulated export failure")
            super().export_to_c(object_file_path, function_name)

    @jit_cache
    def compile_value(value):
        nonlocal compile_count
        compile_count += 1
        return _FlakyArtifact(value)

    with pytest.warns(RuntimeWarning, match="export failed"):
        assert compile_value(53)() == 53
    assert not list(isolated_object_cache.rglob("*.o"))
    assert not list(isolated_object_cache.rglob("*.o.tmp.*"))

    fail_export = False
    compile_value.cache_clear()  # a new process also has no in-memory entry
    assert compile_value(53)() == 53
    assert compile_count == 2
    objects = list(isolated_object_cache.rglob("*.o"))
    assert len(objects) == 1
    assert objects[0].read_text() == "53"


def test_stuck_key_lock_falls_back_with_no_clobber_publish(isolated_object_cache, monkeypatch):
    """A timed-out lock may publish, and a later caller must load that object."""
    from cute_gemm.cache import jit as jit_module
    from cute_gemm.cache import jit_cache

    monkeypatch.setattr(jit_module, "LOCK_TIMEOUT", 0.2)
    compile_count = 0

    @jit_cache
    def compile_value(value):
        nonlocal compile_count
        compile_count += 1
        return _IntArtifact(value)

    sha = jit_module._request_sha(compile_value.__wrapped__, (71,))
    namespace = isolated_object_cache / "stress-namespace"
    namespace.mkdir(parents=True)
    lock_path = namespace / f"{sha}.lock"

    with jit_module.FileLock(lock_path, exclusive=True):
        with pytest.warns(RuntimeWarning, match="lock timeout"):
            assert compile_value(71)() == 71
        assert [p.read_text() for p in namespace.glob("*.o")] == ["71"]

    # The next caller should load the fallback's published object.
    compile_value.cache_clear()
    assert compile_value(71)() == 71
    assert compile_count == 1
    assert [p.read_text() for p in namespace.glob("*.o")] == ["71"]


def test_killed_exporter_does_not_publish_partial_object(tmp_path):
    """A killed writer leaves an orphan temp file; later processes recover."""
    import subprocess
    import sys
    import time

    root = tmp_path / "shared-objects"
    script = tmp_path / "cache_worker.py"
    _CRASH_WORKER = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import cutlass.cute as cute

        class _LoadedModule:
            def __init__(self, path):
                self.path = Path(path)

            def __getitem__(self, name):
                assert name == "func"
                return lambda: int(self.path.read_text())

        cute.runtime.load_module = lambda path, **_: _LoadedModule(path)

        import cute_gemm.cache
        from cute_gemm.cache import jit_cache

        phase = os.environ["TEST_PHASE"]
        ready = Path(os.environ["TEST_READY"])

        class _Artifact:
            def __init__(self, value, phase):
                self.value, self.phase = value, phase

            def export_to_c(self, object_file_path, function_name):
                Path(object_file_path).write_text(str(self.value))
                if self.phase == "crash":
                    # Die mid-export, after the partial write is visible: the
                    # publisher must never run, leaving only the private temp.
                    ready.touch()
                    os._exit(9)

            def __call__(self):
                return self.value

        @jit_cache
        def compile_value(value):
            return _Artifact(value, phase)

        if phase == "crash":
            compile_value(31)
            sys.exit("crash worker unexpectedly survived the export")
        print(f"value={compile_value(31)()}")
        """
    )

    script.write_text(_CRASH_WORKER)
    ready = tmp_path / "partial-written"
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(repo_root),
        DLKERNEL_CACHE_ENABLED="1",
        DLKERNEL_ARTIFACT_CACHE_DIR=str(root),
        TEST_READY=str(ready),
    )

    process = subprocess.Popen(
        [sys.executable, str(script)],
        env={**env, "TEST_PHASE": "crash"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), process.communicate(timeout=5)
    finally:
        process.kill()
        process.communicate(timeout=10)

    assert not list(root.rglob("*.o"))
    assert len(list(root.rglob("*.o.tmp.*"))) == 1

    for phase in ("recover", "warm"):
        result = subprocess.run(
            [sys.executable, str(script)],
            env={**env, "TEST_PHASE": phase},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == "value=31"

    objects = list(root.rglob("*.o"))
    assert len(objects) == 1
    assert objects[0].read_text() == "31"


def test_independent_pools_attribute_one_shared_compile(tmp_path):
    """Two process-local pools may share a root, but only one authored the .o."""
    import json
    import subprocess
    import sys
    import time

    root = tmp_path / "shared-objects"
    markers = tmp_path / "compile-markers"
    markers.mkdir()
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    go = tmp_path / "go"
    _POOL_STUB = textwrap.dedent(
        """
        import functools
        import os
        import time
        from pathlib import Path

        import cutlass.cute as cute

        class Artifact:
            def export_to_c(self, object_file_path, function_name):
                markers = Path(os.environ["TEST_MARKERS"])
                (markers / f"compiled-{os.getpid()}").touch()
                time.sleep(2)  # let the peer observe the same cold key
                Path(object_file_path).write_text("97")

            def __call__(self):
                return 97

        class LoadedModule:
            def __init__(self, path):
                self.path = Path(path)

            def __getitem__(self, name):
                assert name == "func"
                value = int(self.path.read_text())
                return lambda: value

        # Pool workers import this module by name. They must be able to load the
        # placeholder object too if a peer won the flock before they arrived.
        cute.runtime.load_module = lambda path, **kwargs: LoadedModule(path)

        from cute_gemm.cache import jit as jit_module
        from cute_gemm.cache import jit_cache

        # This test isolates pool coordination from package-source edits. Both
        # client processes and their workers import this module before compiling.
        @functools.lru_cache(maxsize=1)
        def stress_fingerprint():
            return "stress-namespace"

        jit_module._compute_source_fingerprint = stress_fingerprint

        @jit_cache
        def compile_value(value):
            return Artifact()
        """
    )

    (tmp_path / "pool_stub.py").write_text(_POOL_STUB)
    client = tmp_path / "pool_client.py"
    _POOL_CLIENT = textwrap.dedent(
        """
        import json
        import os
        import time
        from pathlib import Path

        from cute_gemm.cache import precompile
        from pool_stub import compile_value

        ready = Path(os.environ["TEST_READY_DIR"]) / os.environ["TEST_CLIENT_ID"]
        ready.touch()
        go = Path(os.environ["TEST_GO"])
        deadline = time.monotonic() + 60
        while not go.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("barrier timeout")
            time.sleep(0.01)

        result = precompile([lambda: compile_value(97)])
        print(json.dumps({"pid": os.getpid(),
                          "n_unique": result.n_unique,
                          "n_compiled": result.n_compiled,
                          "n_hits": result.n_hits}), flush=True)
        """
    )

    client.write_text(_POOL_CLIENT)
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(repo_root), str(tmp_path))),
        DLKERNEL_CACHE_ENABLED="1",
        DLKERNEL_ARTIFACT_CACHE_DIR=str(root),
        DLKERNEL_ARCH="90",
        CUTE_DSL_ARCH="sm_90a",
        CUDA_VISIBLE_DEVICES="",
        TEST_MARKERS=str(markers),
        TEST_READY_DIR=str(ready_dir),
        TEST_GO=str(go),
    )
    clients = [
        subprocess.Popen(
            [sys.executable, str(client)],
            env={**env, "TEST_CLIENT_ID": str(index)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(2)
    ]
    try:
        deadline = time.monotonic() + 60
        while len(list(ready_dir.iterdir())) < 2 and time.monotonic() < deadline:
            if any(p.poll() is not None for p in clients):
                break
            time.sleep(0.05)
        assert len(list(ready_dir.iterdir())) == 2
        go.touch()
        outputs = [p.communicate(timeout=120) for p in clients]
    finally:
        for process in clients:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

    reports = []
    for process, (stdout, stderr) in zip(clients, outputs):
        assert process.returncode == 0, stdout.decode() + stderr.decode()
        reports.append(json.loads(stdout.decode().strip().splitlines()[-1]))
    assert sorted(report["n_compiled"] for report in reports) == [0, 1], reports
    assert sorted(report["n_hits"] for report in reports) == [0, 1], reports
    assert all(report["n_unique"] == 1 for report in reports)
    compile_markers = list(markers.iterdir())
    assert len(compile_markers) == 1
    assert int(compile_markers[0].name.removeprefix("compiled-")) not in {
        report["pid"] for report in reports
    }, "compile ran in a client instead of a pool worker"
    assert [p.read_text() for p in root.rglob("*.o")] == ["97"]


# ---------------------------------------------------------------------------
# R4: register_fake schema-split is auto-derived from the eager signature
# ---------------------------------------------------------------------------


def test_parse_concat_layout_roundtrip():
    """``_parse_concat_layout`` is the shared str↔tuple helper used by both
    eager bodies and the fake registration. A regression that breaks one
    side without the other reintroduces the 7acaadd compile-key drift.
    """
    from cute_gemm.gemm_interface import _parse_concat_layout

    assert _parse_concat_layout(None) is None
    assert _parse_concat_layout("") is None  # empty string → None
    assert _parse_concat_layout("a") == ("a",)
    assert _parse_concat_layout("a,b") == ("a", "b")
    # Idempotency: pre-converted tuples pass through unchanged so callers can
    # chain through the helper without checking the input type first.
    assert _parse_concat_layout(("a", "b")) == ("a", "b")
    assert _parse_concat_layout(()) == ()


def test_merge_tensor_helper():
    """Shared ``Union[scalar, Tensor]`` schema-split merge."""
    from cute_gemm.gemm_interface import _merge_tensor

    assert _merge_tensor(1.0, None) == 1.0
    assert _merge_tensor(1.0, "T") == "T"  # non-None tensor_value wins
    assert _merge_tensor(None, "T") == "T"
    assert _merge_tensor(None, None) is None  # both None → stay None


# ---------------------------------------------------------------------------
# Smoke: module imports / re-imports stay consistent
# ---------------------------------------------------------------------------


def test_materialize_stall_guard_fires_on_wedged_worker(
    isolated_object_cache, tmp_path, monkeypatch
):
    """A wedged compile worker trips the rotation loop's stall guard.

    Wedge = the pool pretends every submit is deferred, poll never resolves,
    and no artifact lands (precompile.py: "stall guard against a wedged
    worker" — orthogonal to the per-key flock timeout covered elsewhere).
    Teeth: the guard fires after ~stall_timeout_s (not the 600s default),
    names the outstanding count, publishes no partial .o, and never falls
    back to an in-process compile. Negative control: the same spec with a
    live pool materializes normally under the same tight timeout.
    """
    import time as _time
    from pathlib import Path

    import cute_gemm.cache
    from cute_gemm.cache import jit_cache
    from cute_gemm.cache.async_compile import CompilePool
    from cute_gemm.cache.capture import CompileCaptured, capture_compiles
    from cute_gemm.cache.precompile import materialize

    compiled = []

    @jit_cache
    def _stub_kernel(key):
        compiled.append(key)
        return _PlaceholderCompiled()

    with capture_compiles() as captured:
        with pytest.raises(CompileCaptured):
            _stub_kernel("wedge")
    specs = captured.tasks()
    assert len(specs) == 1

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CompilePool, "submit", lambda self, *a, **k: True)
        mp.setattr(CompilePool, "poll", lambda self, sha: ("pending", None))
        t0 = _time.monotonic()
        with pytest.raises(RuntimeError, match=r"materialize stalled.*1 outstanding"):
            materialize(specs, stall_timeout_s=0.5)
        dt = _time.monotonic() - t0
    assert 0.4 <= dt <= 5.0, f"guard fired at {dt:.2f}s — outside the stall window"
    assert compiled == [], "wedge must not fall back to an in-process compile"
    assert not list(tmp_path.rglob("*.o")), "no partial artifact may be published"

    # Negative control: live pool, same tight timeout — guard stays silent.
    n = materialize(specs, stall_timeout_s=0.5)
    assert n == 1 and compiled == ["wedge"]
