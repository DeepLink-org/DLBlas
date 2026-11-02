# cute_gemm

High-performance CUDA kernels written in
[CuTe-DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL).

## Installation

The distribution is not published to PyPI yet, so install it from a checkout:

``` bash
git clone <your-fork-url> && cd cute_gemm
pip install -e '.'

# With the optional extras
pip install -e '.[cu13]' --extra-index-url https://download.pytorch.org/whl/cu132
pip install -e '.[heuristics]'
```

## Requirements

- H100, B200/B300, 
- CUDA toolkit 12.8+
- Python 3.12

## Kernels

- RMSNorm forward + backward
- Softmax forward + backward
- Cross entropy forward + backward
- Layernorm forward + backward
- Hopper GEMM + epilogue
- Blackwell GEMM + epilogue
- Blackwell GeForce GEMM + epilogue

## Usage

```python
from cute_gemm import rmsnorm, softmax, cross_entropy
```

## Tune modes

GEMM entries (`gemm`, `gemm_add`, epilogue calls, the fused MoE primitives,
`linear`/`mlp`, the fused XTuner adapter) take a per-call `tune_mode` keyword.
It decides what a tune-cache miss may do:

| Mode | A miss |
|---|---|
| `TUNED` (default) | Sweep on every new shape — the pre-TuneMode behaviour |
| `RUNNING` | Measure once per operator contract, then reuse that winner for new token buckets of the same contract |
| `STATIC` | Never measure; reuse the contract's winner if one exists, otherwise run the analytical heuristic |

A miss resolves in order: this shape's measured winner (in memory, then on
disk) → the operator contract's measured winner, transferred across token
buckets → mode-dependent handling above. Measured winners are shared across
all three modes, and a winner another process publishes is picked up on the
next call.

```python
from cute_gemm import TuneMode, gemm

out = gemm(a, b, cu_seqlens_m=cu)                       # default: TUNED
out = gemm(a, b, cu_seqlens_m=cu, tune_mode=TuneMode.RUNNING)  # framework steady state
out = gemm(a, b, cu_seqlens_m=cu, tune_mode=TuneMode.STATIC)   # never measure (prewarm first)
```

The library default is deliberately `TUNED` (the original behaviour). A
framework that wants "measure once per contract" opts in explicitly — see the
XTuner MoE adapter, whose own default is `RUNNING`.

Useful environment variables: `DLKERNEL_REQUIRE_TUNE_CACHE=1` (error instead
of auto-selecting without a measured winner), `DLKERNEL_FORCE_CACHE_UPDATE=1`
(re-sweep despite a disk entry), `DLKERNEL_DEBUG_TUNE=1` (trace misses,
winner loads, reuse). The offline tooling (`python -m cute_gemm.static_tune`
`study` / `production` / `prefill`) pre-tunes and pre-compiles shapes so
`STATIC` deployments never miss. Details: `docs/modules/static_selection.md`.

## SM budget

`set_num_sms` caps how many SMs persistent-kernel grids and the analytical
heuristic may use (DeepGEMM `DeviceRuntime` semantics):

```python
import cute_gemm as dk

dk.set_num_sms(64)     # cap: heuristic waves and persistent launch grids follow
print(dk.get_num_sms())
dk.set_num_sms(0)      # 0 restores the device default
```

Values are validated against the physical count when a device is visible
(`0 < n <= physical SMs`); the cap takes whole clusters into the launch grid
and applies immediately to shapes planned after the change.

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
