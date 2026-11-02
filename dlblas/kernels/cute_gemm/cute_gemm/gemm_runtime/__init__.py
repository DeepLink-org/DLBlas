# Copyright (c) 2026, Tri Dao.
"""Generic epilogue-GEMM runtime: identity (digests, refs, registries), the
plan/compile host layer, the ``dlkernel::gemm_epi`` torch op, and autotune.

See docs/notes/epilogue_transform_reorg.md for the layer map. Import submodules
directly (``cute_gemm.gemm_runtime.identity``); this package root stays empty so
the identity leaf can be imported without pulling the host layer.
"""
