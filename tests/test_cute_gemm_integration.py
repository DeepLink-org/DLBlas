# Copyright (c) 2025, DeepLink.
"""Guard tests for the vendored cute_gemm subtree and its dlblas.cute_gemm adapter.

These tests are GPU-free: they only pin down the integration invariants
(subtree shape, packaging isolation, import isolation, forwarding identity).
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBTREE = REPO_ROOT / "dlblas" / "kernels" / "cute_gemm"


def test_subtree_shape():
    """The vendored subtree stays a non-package directory with its own project."""
    assert (SUBTREE / "pyproject.toml").is_file()
    assert (SUBTREE / "LICENSE").is_file()
    assert (SUBTREE / "cute_gemm" / "__init__.py").is_file()
    # A __init__.py here would make find_packages() swallow cute_gemm into dlblas.
    assert not (SUBTREE / "__init__.py").exists()
    # The package was renamed from DLKernel; the old tree must not come back.
    assert not (SUBTREE / "DLKernel").exists()


def test_subtree_not_packaged_into_dlblas():
    """setuptools must never pick up the vendored cute_gemm as a dlblas package."""
    from setuptools import find_packages

    hits = [
        p
        for p in find_packages(where=str(REPO_ROOT))
        if p.startswith("dlblas.kernels.cute_gemm")
    ]
    assert not hits


def test_import_dlblas_does_not_load_cute_gemm():
    """import dlblas must stay CuTe-free: no cute_gemm, no cutlass DSL."""
    code = (
        "import sys, dlblas; "
        "assert 'cute_gemm' not in sys.modules; "
        "assert not any(m == 'cutlass' or m.startswith('cutlass.') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), check=True)


def test_adapter_forwards_to_cute_gemm():
    """dlblas.cute_gemm symbols must be the very objects of the cute_gemm package."""
    import dlblas.cute_gemm as cute

    # The adapter imported the package (installed or in-repo fallback), so the
    # top-level module is registered under its own name.
    impl = sys.modules["cute_gemm"]

    assert cute.__version__ == impl.__version__
    assert cute.gemm is impl.gemm
    assert cute.rmsnorm is impl.rmsnorm
    assert cute.gemm.__module__.startswith("cute_gemm")
    assert cute.GemmSm90 is impl.gemm_sm90.GemmSm90
