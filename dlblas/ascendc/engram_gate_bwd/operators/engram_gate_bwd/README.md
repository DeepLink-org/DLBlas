# EngramGateBwd AscendC Operator (Rev 4.0)

Single-kernel implementation of the EngramGateBwd operator for Ascend NPU (DAV_2201).

## Overview

Computes gradients for the Engram Gate operation: given `grad_out`, `x`, `k`, `v`, `wh`, `we` (all bf16), produces `grad_x`, `grad_k`, `grad_v`, `grad_wh`, `grad_we` (all bf16).

All intermediate values are kept in fp32 within VECCALC, with zero GM intermediate storage.

## Architecture

| Item | Value |
|------|-------|
| Kernel count | 1 (single kernel) |
| T_TILE | 4 |
| Compute position | VECCALC (fp32) |
| Staging position | VECIN (bf16 only) |
| Multi-core | T-dimension split with bf16 atomic add for grad_wh/grad_we |
| NpuArch | dav-2201 (Ascend910B2) |
| CANN | 9.0.0 |

## Key Design Decisions

1. **Double-buffered accumulators** for grad_wh/grad_we to avoid Add(d==a) pattern
2. **Shared 3D term recomputation** after RST overwrite
3. **rstd_k^3 saved** to prevent TC corruption by grad_wh/grad_we
4. **PipeBarrier** at all pipeline boundaries
5. **T_TILE=4 enforced** by limiting blockNum

## Building

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build && cd build
cmake ..
make -j4
```

## Testing

```bash
bash run.sh [T]
# Example: bash run.sh 14
```

## Known Limitations

- **Precision**: Does not meet the 0.00781 MERE target on DAV_2201. The bf16 atomic add for multi-core grad_wh/grad_we introduces ~0.4% per-core quantization error, and the bf16 output cast adds an irreducible precision floor.
- **T_TILE=1 case**: Degraded precision for very small T values where T_per_core < T_TILE.
- **Recommendation**: DAV_3510 + RegBase path recommended for production use.

## File Structure

```
operators/engram_gate_bwd/
├── op_kernel/
│   ├── engram_gate_bwd_tiling.h      # Tiling constants and struct
│   └── engram_gate_bwd_kernel.asc    # Single kernel
├── op_host/
│   ├── engram_gate_bwd.asc           # Host + main entry
│   └── data_utils.h                  # File I/O utilities
├── op_extension/
│   ├── engram_gate_bwd_torch.cpp     # PyTorch extension
│   ├── register.cpp                  # TORCH_LIBRARY registration
│   └── ops.h                         # Function declarations
├── scripts/
│   ├── gen_data.py                   # Test data generation
│   ├── golden.py                     # Reference computation
│   ├── verify_result.py              # Precision verification
│   └── test_torch.py                 # PyTorch integration test
├── CMakeLists.txt
├── run.sh
└── README.md
```

## Reference

- Design document: `docs/DESIGN.md`
- Development plan: `docs/PLAN.md`
- PyTorch reference: `/mnt/data01/zmz/workspace/12agent/waic/origin/engram_gate_bwd.py`
