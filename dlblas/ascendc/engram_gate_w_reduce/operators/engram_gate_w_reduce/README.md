# engram_gate_w_reduce

Fused Reduction + Broadcast Multiply-Accumulate operator for Ascend NPU (DAV_2201 / Ascend 910B2).

## Operator Definition

```
grad_w_sum = sum(grad_w_partial, dim=0)           // [108, 4, H] → [4, H]
grad_weight_hidden += grad_w_sum ⊙ weight_embed   // broadcast mul-add
grad_weight_embed += grad_w_sum ⊙ weight_hidden   // broadcast mul-add
```

## Inputs

| # | Name | Shape | dtype |
|---|------|-------|-------|
| I1 | grad_w_partial | [108, 4, hidden_size] | float32 |
| I2 | weight_hidden | [4, hidden_size] | bfloat16 |
| I3 | weight_embed | [4, hidden_size] | bfloat16 |
| I4 | grad_weight_hidden | [4, hidden_size] | float32 (in-place) |
| I5 | grad_weight_embed | [4, hidden_size] | float32 (in-place) |

## Outputs

| Name | Shape | dtype |
|------|-------|-------|
| grad_weight_hidden | [4, hidden_size] | float32 |
| grad_weight_embed | [4, hidden_size] | float32 |

## Quick Start

```bash
# Build and test with default hidden_size=4096 (direct invoke)
bash run.sh

# Test with custom hidden_size (direct invoke)
HIDDEN_SIZE=1024 bash run.sh --skip-build

# PyTorch integration test
cd build
python3 ../scripts/test_torch.py --hidden_size 4096
```

## Test Results

All test cases pass with FP32 perfect match (max_diff = 0.0).

| hidden_size | Direct Invoke | PyTorch |
|-------------|---------------|---------|
| 1 | PASSED | - |
| 4 | PASSED | - |
| 13 | PASSED | - |
| 64 | PASSED | - |
| 256 | PASSED | - |
| 1024 | PASSED | - |
| 4096 | PASSED | PASSED |
| 8192 | PASSED | - |

## Performance

Kernel execution time for hidden_size=4096: **186.3 us** (on Ascend 910B2, 48 cores).

The kernel is memory-bound (MTE2 = 83.9% of AIV time), dominated by Phase 1 row-by-row data loading (108 rows x 4 channels = 432 DMA transfers).

## Architecture

- **Platform**: DAV_2201 (Ascend 910B2), CANN 9.0.0
- **Programming Model**: AscendC SIMD / MemBase
- **Fusion**: Single kernel: Phase 1 (reduction) + Phase 2 (multiply-accumulate)
- **Multi-core**: Split along hidden_size dimension
- **Synchronization**: PipeBarrier for DMA/compute ordering

## Files

| Path | Description |
|------|-------------|
| `op_kernel/engram_gate_w_reduce_tiling.h` | Tiling structure (kernel + host shared) |
| `op_kernel/engram_gate_w_reduce_kernel.asc` | Kernel implementation |
| `op_host/engram_gate_w_reduce.asc` | Host entry + main |
| `op_host/data_utils.h` | File I/O utilities |
| `op_extension/ops.h` | PyTorch extension function declarations |
| `op_extension/engram_gate_w_reduce_torch.cpp` | PyTorch extension implementation |
| `op_extension/register.cpp` | TORCH_LIBRARY registration |
| `scripts/gen_data.py` | Test data generator |
| `scripts/golden.py` | Reference golden computation |
| `scripts/verify_result.py` | Accuracy verification |
| `scripts/test_torch.py` | PyTorch integration test |
| `CMakeLists.txt` | Build configuration (dual target) |
| `run.sh` | One-click build & test script |
| `docs/DESIGN.md` | Technical design document |
| `docs/PLAN.md` | Development plan & results |
| `docs/perf/round_001/` | Profiling data (round 1) |
| `docs/perf/round_002/` | Profiling data (round 2) |
