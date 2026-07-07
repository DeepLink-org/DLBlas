# head_compute_mix_bwd

## Overview

Ascend C implementation of `mhc_head_compute_mix` backward pass. Fuses Broadcast + Elementwise + Reduction computations into a single kernel with multi-core Group Reduce.

- **Architecture**: DAV_2201 (Ascend910B2)
- **CANN**: 9.0.0
- **Route**: SIMD/MemBase (Vector API)
- **Precision**: float32 only
- **Standard shape**: (2, 1024, 4)

## Mathematical Definition

```
Input:  input_mix(B,S,C), mhc_scale(1), mhc_base(C), grad_out(B,S,C)
Output: grad_input_mix(B,S,C), grad_mhc_scale(1), grad_mhc_base(C)

z               = input_mix * mhc_scale + mhc_base
sigmoid         = 1 / (1 + exp(-z))
sigmoid_grad    = sigmoid * (1 - sigmoid)
grad_z          = grad_out * sigmoid_grad
grad_input_mix  = grad_z * mhc_scale
grad_mhc_base   = sum(grad_z, dim=(0,1))
temp            = grad_z * input_mix
grad_mhc_scale  = sum(temp, dim=(0,1,2))
```

## Quick Start

### Prerequisites

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
```

### Build and Run

```bash
# Full build + standard test
bash run.sh

# Skip build (reuse artifacts)
bash run.sh --skip-build

# Comprehensive test suite (direct invoke + PyTorch)
bash run.sh --all

# PyTorch pathway only
bash run.sh --torch
```

### Direct Invoke (Executable)

```bash
cd build
python3 ../scripts/gen_data.py 2 1024 4    # (n0, n1, mhc_mult)
./head_compute_mix_bwd 2 1024 4
python3 ../scripts/verify_result.py
```

### PyTorch Pathway

```python
import torch
import torch_npu

torch.ops.load_library("build/libhead_compute_mix_bwd_ops.so")

B, S, C = 2, 1024, 4
im = torch.randn(B, S, C, dtype=torch.float32).npu()
ms = torch.randn(1, dtype=torch.float32).npu()
mb = torch.randn(C, dtype=torch.float32).npu()
go = torch.randn(B, S, C, dtype=torch.float32).npu()

g1, g2, g3 = torch.ops.npu.head_compute_mix_bwd(im, ms, mb, go)
```

## File Structure

```
├── op_kernel/
│   ├── head_compute_mix_bwd_tiling.h       # TilingData struct + constants
│   └── head_compute_mix_bwd_kernel.asc     # Kernel (KernelHeadComputeMixBwd class)
├── op_host/
│   ├── head_compute_mix_bwd.asc            # Host entry + main (CLI shape params)
│   └── data_utils.h                        # Binary file read/write
├── op_extension/
│   ├── head_compute_mix_bwd_torch.cpp       # PyTorch extension (stream(true) sync)
│   ├── register.cpp                        # TORCH_LIBRARY (PrivateUse1 + Meta)
│   └── ops.h                               # Function declarations
├── scripts/
│   ├── gen_data.py                         # Test data generation (CLI params)
│   ├── golden.py                           # NumPy reference computation
│   ├── verify_result.py                    # Single-case precision verification
│   ├── test_torch.py                       # PyTorch comprehensive test (12 cases)
│   └── test_all.py                         # Direct invoke comprehensive test (12 cases)
├── CMakeLists.txt                          # Dual target: exe + libhead_compute_mix_bwd_ops.so
├── run.sh                                  # Build + run script
└── docs/
    ├── DESIGN.md                           # Architecture design document
    ├── PLAN.md                             # Development plan + results
    └── perf/round_002/                     # Performance profiling data
```

## Design Summary

| Aspect | Decision |
|--------|----------|
| Kernel strategy | Single kernel fusion with multi-core Group Reduce |
| UB layout | 4-buffer scheme: inQIm(DB), inQGo(DB), bufZ, outQOut(DB) |
| Data movement | DataCopyPad throughout (no 32B alignment constraint) |
| Multi-core | Rows split across cores; partial reductions per core; Core 0 merges via workspace |
| Broadcast | mhc_base[4] expanded to 8-element pattern for BinaryRepeatParams alignment |
| Reduction | Column-wise manual accumulation (inner_dim=4) + ReduceSum API for full reduction |
| Synchronization | PipeBarrier<PIPE_V>() before SyncAll(); Core 0 exclusive merge after barrier |
| Stream sync | `stream(true)` (clear queue) for PyTorch pathway to prevent out-of-order execution |

## Test Results

### Direct Invoke (12 tests, all passed)

| Category | Test | Shape | Max Diff |
|----------|------|-------|----------|
| FT | Standard | 2x1024x4 | 4.29e-06 |
| FT | Minimum | 1x1x4 | 2.24e-08 |
| FT | Asymmetric | 4x512x4 | 1.14e-05 |
| FT | Large n1 | 2x4096x4 | 9.54e-05 |
| FT | Random | 3x2048x4 | 9.54e-06 |
| BT | Zeros input | 2x1024x4 | 1.53e-05 |
| BT | input_mix=+100 | 2x1024x4 | 1.19e-07 |
| BT | input_mix=-100 | 2x1024x4 | 3.81e-06 |
| BT | mhc_scale=0 | 2x1024x4 | 8.11e-06 |
| BT | Diverse bias | 2x1024x4 | 2.86e-06 |

### PyTorch Pathway (12 tests, all passed)

All 12 test cases (FT-01~05, BT-01~05, L0 tiny) passed with max diff < 8.01e-05.

**Precision standard**: rtol=1e-4, atol=1e-6 (float32 community standard)

## Performance

| Metric | Value |
|--------|-------|
| Kernel time | **12.54 us** (msprof op, 5-launch average) |
| Cores used | 8 |
| Data volume | ~100 KB total |
| Sigmoid tmpBuf | 32 KB |
| Head overhead | 2.61 us (20.8%) |
| Dominant pipeline | SCALAR (75.65%) |
| Vec compute | 6.32% |
| MTE2/MTE3 | 10.19%/4.22% |

### Performance Notes

- This is an ultra-lightweight fused operator (input 32KB, ~182K FLOPs)
- SCALAR dominance is inherent to the Group Reduce design (SyncAll barrier + Core 0 merge)
- The 12.54 us execution time is near the hardware minimum for this class of operator
- Profiling data archived at `docs/perf/round_002/`
