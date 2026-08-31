from __future__ import annotations

import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401


_LOADED = False


def load_library() -> None:
    global _LOADED
    if _LOADED:
        return

    override = os.getenv("DLBLAS_KS_ASCENDC_LIBRARY")
    candidates = []
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent
    candidates.extend(
        [
            root / "build" / "libdlblas_ks_ascendc_ops.so",
            root / "lib" / "libdlblas_ks_ascendc_ops.so",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            torch.ops.load_library(str(candidate))
            _LOADED = True
            return

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "DLBlas KernelSwift AscendC library was not found. "
        f"Searched: {searched}. Run ascend/clike_910b/build.sh first."
    )
