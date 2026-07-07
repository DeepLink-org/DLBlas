# BigFuse — MHC Pre-processing Fused Operator

Multi-Head Composition (MHC) pre-processing fused into a single 3-kernel Ascend C pipeline.

## Overview

BigFuse combines four MHC stages into one operator:

```
S1: RMS-Normalized Linear Projection  (K0 + K1)
S2: Split Mixes + Sigmoid             (K2)
S3: Sinkhorn Doubly-Stochastic Norm   (K2)
S4: Weighted Sum (Apply Mix)          (K2)
```

**Architecture**: 3-Kernel pipeline optimized for DAV_2201 (Ascend910B2)

| Kernel | Type | Cores | Function |
|--------|------|-------|----------|
| K0 | AIV (Vector) | 48 | bf16 residual -> fp32 flat conversion |
| K1 | AIC (Cube) | 8 | Linear projection via MatMul |
| K2 | AIV (Vector) | 43 | RMS Norm + Sigmoid + Sinkhorn + Apply Mix |

## Performance

| Metric | Value |
|--------|-------|
| AscendC total | 1.866 ms |
| PyTorch CPU | 53.27 ms |
| Speedup | **28.55x** |
| Precision | All outputs pass (fp32 MERE < 1e-3, bf16 abs < 2 ULP) |

### Per-kernel Breakdown

| Kernel | Time | % of Total | Bottleneck |
|--------|------|-----------|------------|
| K0 (bf16->fp32) | 403 us | 21.6% | Memory-bound |
| K1 (MatMul) | 338 us | 18.1% | MTE2 95%, Cube util 19% |
| K2 (Post-process) | 1126 us | 60.3% | Scalar-bound 99.8% |

## Input/Output

| Tensor | Shape | DType | Role |
|--------|-------|-------|------|
| residual (in) | [B, S, M, H] = [1, 512, 4, 1280] | bf16 | Input residual |
| fn_weight (in) | [K, D] = [24, 5120] | fp32 | Projection weight |
| mhc_scale (in) | [3] | fp32 | Scale factors |
| mhc_base (in) | [K] = [24] | fp32 | Bias values |
| post_mix (out) | [B, S, M, 1] | fp32 | Post mixing weights |
| comb_mix (out) | [B, S, M, M] | fp32 | Combination matrix |
| layer_input (out) | [B, S, H] | bf16 | Weighted layer input |

## Build & Run

### Prerequisites

- CANN 9.0.0
- Ascend910B2 (DAV_2201)
- CMake >= 3.16
- bisheng compiler

### Quick Start

```bash
# Build, generate test data, compute golden, run kernels, verify
cd operators/big_fuse
bash run.sh

# Skip rebuild (use existing artifacts)
bash run.sh --skip-build
```

### Build PyTorch Extension (experimental)

```bash
cd operators/big_fuse
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
# Produces: build/libbig_fuse_ops.so
```

Note: The PyTorch extension builds successfully but has a runtime hang on kernel launch. Use the direct-invoke path (`bash run.sh`) for production.

## Project Structure

```
operators/big_fuse/
|-- CMakeLists.txt              # Dual-target: executable + .so
|-- run.sh                      # Build & test script
|-- op_kernel/                   # Device kernels
|   |-- big_fuse_k0.asc          # K0: bf16->fp32 conversion (AIV)
|   |-- big_fuse_k1.asc          # K1: MatMul (AIC)
|   |-- big_fuse_k2.asc          # K2: Vector post-process (AIV)
|-- op_host/
|   |-- big_fuse.asc             # Host: tiling + 3-kernel launch + main
|   |-- data_utils.h             # Binary I/O utilities
|-- op_extension/                # PyTorch TORCH_LIBRARY (experimental)
|   |-- big_fuse_torch.cpp
|   |-- register.cpp
|   |-- ops.h
|-- tiling/
|   |-- big_fuse_tiling.h        # TilingHeaderK0/K1/K2 definitions
|-- scripts/
|   |-- gen_data.py              # Test data generation
|   |-- golden.py                # PyTorch golden computation
|   |-- verify_result.py         # Precision verification
|   |-- benchmark.py             # Performance comparison
|   |-- test_multilevel.py       # Multi-level tests
|   |-- test_torch.py            # PyTorch extension test
|-- docs/
|   |-- DESIGN.md                # Architecture design
|   |-- PLAN.md                  # Development plan & status
|   |-- environment.md           # Environment info
|   |-- perf/                    # Performance data archives
|       |-- round_001/           # Initial profile
|       |-- round_002/           # K1/K2 per-kernel profile
|       |-- round_003/           # Full msprof op analysis
|       |-- round_004/           # Current benchmark
```

## Tiling Configuration

| Parameter | K0 | K1 | K2 |
|-----------|----|----|-----|
| Core type | AIV (48) | AIC (8) | AIV (43) |
| tokensPerCore | 11 | - | 12 |
| tokensPerTile | 4 | - | 2 |
| singleCoreM | - | 64 | - |
| singleCoreN | - | 24 | - |

K2 uses T=2 (not T=3 as in DESIGN) because T=3 produces 48-byte post_mix writes that violate DataCopyPad's 32-byte alignment requirement.

## Precision

| Output | DType | MERE Threshold | Measured | Status |
|--------|-------|---------------|----------|--------|
| post_mix | fp32 | 9.77e-04 (2^-10) | 3.90e-04 | PASS |
| comb_mix | fp32 | 9.77e-04 (2^-10) | 8.55e-04 | PASS |
| layer_input | bf16 | 2 ULP (1.56e-02) | 7.81e-03 | PASS |

## Known Limitations

1. **K2 scalar-bound**: sqrsum accumulation and Sinkhorn normalization use GetValue/SetValue scalar loops (99.8% scalar). Vectorizing with BlockReduceSum would significantly improve K2 performance.
2. **PyTorch extension runtime hang**: The TORCH_LIBRARY .so builds but hangs on kernel launch. Likely a stream(true) + function-call convention issue in the ASC/C++ interop.
3. **K2 T=3 not viable**: 48-byte post_mix writes are not 32B-aligned for DataCopyPad. T=2 is the practical maximum with current UB budget.
4. **Shape fixed**: Current implementation hardcodes S=512, M=4, H=1280. The TilingHeader design supports arbitrary shapes, but the host code and data generation need generalization.

## References

- [DESIGN.md](docs/DESIGN.md) — Full architecture design
- [PLAN.md](docs/PLAN.md) — Development plan and status
- [environment.md](docs/environment.md) — Environment configuration
