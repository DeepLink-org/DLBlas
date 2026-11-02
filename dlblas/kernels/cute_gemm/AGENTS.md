This file provides guidance to coding agents (Claude Code, Codex) when working with code in this repository.

## Project Overview

cute_gemm — high-performance CUDA kernels written in [CuTe-DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL), targeting H100 (SM90), B200/B300 (SM100), and GeForce RTX 50 (SM120) GPUs. Package name: `cute_gemm`.

## Build & Development

```bash
# Install (dev)
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.x
pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Lint & format
ruff check --fix cute_gemm/ tests/ benchmarks/
ruff format cute_gemm/ tests/ benchmarks/

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_rmsnorm.py -x
pytest tests/test_rmsnorm.py::test_rmsnorm_fwd -x -k "bfloat16"

# After editing kernel source (cold .o cache): overlap compiles with tests.
# Misses are compiled by a pool of N CPU workers (forkserver sidecar) while
# deferred tests retry once their .o lands. No-op when the cache is warm.
pytest tests/test_rmsnorm.py --async-compile=16
pytest tests/ -n 8 --async-compile=32
```

## CuTe DSL Conventions

For code inside `@cute.jit` or `@cute.kernel` decorated functions, only a subset of Python syntax is supported. Follow [Control Flow](docs/dsl_control_flow.rst) and [Limitations](docs/limitations.rst).

Key rules:
- `cutlass.const_expr()` marks compile-time constants; `cutlass.range_constexpr()` unrolls loops at compile time
- `cutlass.range()` is for dynamic runtime loops
- No early `break`/`continue` in loops; no `return` values from jit functions
- Python lists/dicts inside DSL are static (compile-time only)
- Types must be determinable at compile time (no dependent types)
- Variables defined inside control flow bodies are not accessible outside

**Note:** CuTe DSL relies on Python source inspection `inspect.getsourcelines()` to parse kernel definitions. Defining `@cute.kernel` / `@cute.jit` functions directly in plain Python REPL will cause source inspection to fail with errors like `OSError: could not get source code`. One should write to a temporary file.

## Architecture

### Kernel patterns

**Reduction kernels** (`rmsnorm.py`, `softmax.py`, `cross_entropy.py`) inherit from `ReductionBase` in `reduction_base.py`. They share a pattern: configure cluster size, get tiled copies, allocate reduction buffers with mbarriers, then launch a `@cute.kernel`.

**GEMM** has a multi-layer design:
- `gemm_interface.py` — variant API on top: unified entry (`gemm`/`gemm_act`/...), validates inputs, allocates outputs
- `gemm.py` — low-level dispatch it calls: plan cache, SM version selection, jit compile + launch
- `gemm_sm90.py` / `gemm_sm100.py` — SM-specific implementations
- `fused_grouped_gemm/` — SM90 MoE fused primitives (role class owning the host contract over the plain SM90 core)
- `gemm_sm90_ptr_array.py` — SM90 pointer-array grouped GEMM: one launch over multiple weight allocations (device `B_ptrs` + per-call tensormap slot workspace); sibling compile wrapper after `gemm_symmetric.py`
- `cute_gemm/epilogue/` — fused epilogues: `ops.py` (EpiOp vocabulary) → `mixin.py`
  (ComposableEpiMixin) → `frontend.py`/`visit.py` (`@gemm_epilogue` fn authoring +
  minted kernel classes) → `library.py` + domain modules (rotary, scaled_exp, ...).
  Hand-written mixins (`gemm_default_epi.py`) are the escape hatch.
- `cute_gemm/gemm_runtime/` — generic host plumbing shared by every epilogue/transform:
  `identity.py` (digests, refs, registries — a leaf) → `host.py` (plan/compile/launch) →
  `torch_op.py` (the single `dlkernel::gemm_epi` custom op) + `autotune.py`
- `cute_gemm/operand_transform/` — A-operand transforms (dequant, dropout, value fns):
  `transform.py`/`kinds.py` (kernel-side) → `frontend.py` (`@a_transform`, handles) →
  `host.py` (bundles, W4 config rules) + `formats/` (packed-weight decode formats)
- `gemm_config.py` — `GemmConfig` dataclass with tile sizes, cluster dims, swizzle settings

### Core utilities

- `copy_utils.py` — memory copy operations (shared↔register, async copies, tiled copies)
- `layout_utils.py` — layout algebra (transpose, select, expand, permute)
- `cute_dsl_utils.py` — dtype mapping, device capability queries, parameter base classes
- `tile_scheduler.py` — tile scheduling for persistent kernels
- `varlen_utils.py` — variable-length sequence support

### Testing

Tests use pytest with parametrize across dtypes (`float32`, `float16`, `bfloat16`), dimensions, and batch sizes. Each test includes a reference implementation for numerical validation.

Every test must verify **numerical correctness** against a reference, not just shapes or smoke.

The public API surface is locked by `tests/test_public_api.py` (exact `__all__`, signature
sequences, keyword-only boundaries) — a deliberate signature change updates that lock first. A test that only checks `.shape` or "doesn't crash" is not a test — it hides bugs. Always compare kernel output values against a PyTorch reference (float32 for ground truth, same dtype for tolerance baseline).

## Deep-dive docs: see docs/README.md

`docs/` holds system documentation only. Specs, plans, and other working
documents live outside it.

## Documentation Sync

A code change that alters behavior, interfaces, or module structure updates
the affected `docs/` pages in the same change. Write those pages per
ASD-STE100. Updating `AGENTS.md` meets a stricter bar: only durable
agent-facing facts — build and test commands, the architecture map,
hard-won gotchas. No changelogs. No per-commit records. No speculative
guidance. When in doubt, leave `AGENTS.md` unchanged.

## Iteration Speed

When iterating on kernel code, run a small subset of tests (1-3 parametrizations) rather than the full test suite. Use `-k` or pass specific test IDs to pytest. Only run the full suite when finalizing changes.

## Debugging Failures

Find out why a failure happens before you fix it. Making the test pass without understanding the cause hides the bug.

Do not route around a bug to make a test pass — no pruning configs, skipping paths, resetting state, raising limits, or swapping implementations. A workaround that makes CI green but leaves the bug hides the problem; it will resurface at greater cost. Accept a workaround only after you prove the root cause is external, and record why.

Start by reproducing the reported failure, then simplify it to the smallest setting that still fails: reduce batch, M/N/K, tile shape, scheduler options, `swap_ab`, beta/C, dtype, and epilogue features where possible. Keep the failing behavior intact while removing unrelated complexity.

Use `cute.printf` inside `@cute.jit` / `@cute.kernel` code to print the relevant locations, tile coordinates, tensor coordinates, predicates, and values. Print at the boundaries between stages, for example TMA load, MMA accumulator, epilogue register values, register-to-smem, smem contents, and TMA store coordinates, until the first bad stage is identified.

After finding a fix, verify that the minimized repro passes, the original repro passes, and that temporarily disabling the fix makes the regression test fail. Regression tests should encode the failure mode, not only the high-level symptom.

## Code Style

- Favor concise, self-explanatory code
- Line length: 100 (ruff)
- Ruff allows: lambda assignment (E731), single-char vars I/O/l (E741), unused locals (F841)
- Cache stability is NOT a design constraint: never contort code (epilogue tags,
  jit-cache keys, semantic digests) to keep compiled-kernel or autotune caches warm
  across a refactor. A one-time recompile/retune is fine; cleaner code wins.

## Measured facts & gotchas (hard-won; verify before assuming they changed)

Hardware (B300 / SM100):
- Dense bf16 peak is 2250 TFLOPS — never use the sparse marketing number.
- L1TEX: gmem and smem share the 128 B/clk/SM pipe; 16-bit kernels with smem
  exchanges are smem/latency-bound, not DRAM-bound (clone ceiling ~6.3 TB/s).
- Blockscaled SF tmem is 64-N granular at both UTCCP write and QMMA read:
  tile_n that is a 32-multiple but not 64 (96, 224) is hardware-blocked.

Benchmarking (see tools/matmul_heuristic/common.py for the reference protocol):
- Per-launch CUDA event pairs on a starved stream time the host enqueue gap,
  not the kernel — any sub-100us kernel needs a GPU-side backlog (burner).
  Consecutive kernels chain L2 state, so rotate config order across rounds.
- On shared nodes: interleave variants at single-launch granularity and
  compare medians; sequential do_bench rounds drift 2-3x with clocks and
  co-tenants. Guard long benches with a contention canary.
- Short benches ride boost clocks (~15-25% fast vs settled); input data
  distribution shifts the settled clock (randn vs small-int vs zeros differ
  up to ~40%) — match warmup AND data distribution across baselines.
- Pin the CPU (taskset) when launch overhead matters; the Python/FFI launch
  path floor is ~3.5us and drifts 2-3x unpinned.

CuTe-DSL:
- Dtype views of swizzled smem drop the swizzle: re-attach the SAME
  byte-addressed swizzle via `recast_ptr`, never plain `recast_tensor`.
- TensorSSA `.store` on all-zero-stride filtered rmem views is silently
  dropped — use scalar setitem.
- Data-independent per-element selects hoist above the load and spill
  registers; make both select arms consume the loaded value.
- Never `partition_D` a tile smaller than the epilogue tiler (warp
  corruption); derive extents from the copy atom.
- `DLKERNEL_CACHE_ENABLED=0` for const_expr ablations — the jit cache ignores
  env flags, and identical timings across an ablation mean a stale cubin.
- `@autotune` persists tune results to `~/.dlkernel/cache` by default
  (`cache_results=True` at the decorator): warm calls execute only the best
  config — tests that must sweep force it cold (`cache_results=False`).
- ruff format >= 0.16 rewrites `Optional[X]` in DSL signatures to `X | None`;
  both trace identically, but fork kernel code verbatim and re-verify.
