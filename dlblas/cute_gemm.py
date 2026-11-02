# Copyright (c) 2025, DeepLink.
"""Lazy forwarding adapter for the vendored cute_gemm (CuTe-DSL) operator library.

cute_gemm is vendored under ``dlblas/kernels/cute_gemm`` and keeps its own
package identity. It is imported as the top-level ``cute_gemm`` package; when
it is not installed, this adapter falls back to the in-repo copy. This module
only re-exports cute_gemm's public API under the ``dlblas.cute_gemm``
namespace; operator implementations, autotuning and caches remain entirely
cute_gemm's own and are never wired into DLBlas's Triton OpRegistry or cache.
"""

import importlib
import sys
from pathlib import Path

try:
    import cute_gemm as _impl
except ModuleNotFoundError as exc:
    if exc.name != "cute_gemm":
        raise
    _vendored = Path(__file__).resolve().parent / "kernels" / "cute_gemm"
    if not (_vendored / "cute_gemm" / "__init__.py").is_file():
        raise ImportError(
            "cute_gemm is not installed. Run: "
            "pip install -e ./dlblas/kernels/cute_gemm"
        ) from exc
    sys.path.insert(0, str(_vendored))
    import cute_gemm as _impl

__version__ = _impl.__version__
__all__ = [*_impl.__all__, "__version__"]

for _name in _impl.__all__:
    globals()[_name] = getattr(_impl, _name)

# Internals used by DLBlas integration tests; not part of the public API.
from cute_gemm.epilogue import library as _epi_library  # noqa: E402
from cute_gemm.gemm_base import GemmBase  # noqa: E402
from cute_gemm.gemm_sm90 import GemmSm90  # noqa: E402

linear_epi = _epi_library.linear_epi
relu_mod = _epi_library.relu_mod
scaled_residual = _epi_library.scaled_residual


def __getattr__(name):
    """Forward any other attribute, including submodules, to cute_gemm."""
    try:
        return getattr(_impl, name)
    except AttributeError:
        module_name = f"{_impl.__name__}.{name}"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only "no such submodule" becomes AttributeError; a missing
            # dependency inside an existing submodule must stay visible.
            if exc.name != module_name:
                raise
            raise AttributeError(name) from None


del _name, _epi_library
