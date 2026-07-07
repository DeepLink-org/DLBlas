# C500 Trace Profile Digest: optimized

## Kernel

- Name: `_Z30engram_fused_weight_kernel_optPKDF16_S0_Pfi`
- Grid: `[2, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `4,149,731` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 36.92% / 5.25%
- GVM share / VLS duty / L2C duty: 4.62% / 0.00% / 0.00%
- VL1/L2C hit rate: 96.82% / 29.75%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 1.00%
- WIF P25/P50/P75/max: 1.00 / 2.00 / 3.00 / 4
- DPC balance: min/avg/max=9,216/11,253/13,016 cycles, imbalance=33.77%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 4,149,731 | cycles | lower |
| CycleTrace instructions | 130 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 36.92 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 4.615 | % | lower |
| CycleTrace ARRIVE share | 10.77 | % | lower |
| CycleTrace LDU share | 6.154 | % | context |
| CycleTrace NOP share | 12.31 | % | lower |
| Wave in flight mean | 2 | waves | higher |
| Wave in flight max | 4 | waves | higher |
| Wave in flight P25 | 1 | waves | context |
| Wave in flight P50 | 2 | waves | context |
| Wave in flight P75 | 3 | waves | context |
| GVM issue peak / 600 cycles | 3 | inst | lower |
| Registers/thread | 6 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 1 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 1 | % | higher |
| AP MTE duty | 5.247 | % | higher |
| AP STE duty | 3.676 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 4.64 | inst/cycle | higher |
| Instruction throughput efficiency | 1.115 | % | higher |
| Instructions per AP | 491.5 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 52.28 | % | lower |
| ISU stall cycles | 7,175 | cycles | lower |
| VL1 hit rate | 96.82 | % | higher |
| L2C hit rate | 29.75 | % | higher |
| DNOC read requests | 5,427 | req | lower |
| DNOC write requests | 2,915 | req | lower |
| Global memory read bytes | 440,896 | bytes | lower |
| Global memory write bytes | 253,056 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 848 | waves | higher |
| Dispatched waves | 856 | waves | context |
| Average wave life | 605.4 | cycles | lower |
| Empirical achieved FLOPS | 0.1433 | TFLOPS | higher |
| Empirical achieved bandwidth | 70.86 | GB/s | higher |
| Empirical roofline intensity | 2.023 | FLOP/Byte | context |
| Roofline HBM usage | 3.845 | % | higher |
| Roofline VL1 usage | 0.4367 | % | higher |
| Roofline L2C usage | 0.472 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 2.191 | FLOP/Byte | context |
| L2C-level roofline intensity | 6.59 | FLOP/Byte | context |
| DPC compute imbalance | 33.77 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 48 | 36.92% | Vector/Matrix Tensor Extension compute category |
| STE | 54 | 41.54% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 6 | 4.62% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 14 | 10.77% | Synchronization/arrival category |
| LDU | 8 | 6.15% | Load data unit category |

- STE subevents by `name`: STE=36, S_NOP=16, Branch=2.
- GLOBAL subevents by `name`: GVM Load=4, GVM Store=2.
- ARRIVE subevents by `name`: Synchronization=14.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 5.25% / 3.68% / 0.00%
- Real IPC: 4.64
- VL1/L2C hit rate: 96.82% / 29.75%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=9,216/11,253/13,016 cycles, imbalance=33.77%
- Achieved/dispatched waves: 848 / 856; workgroups=214; avg wave life=605.4 cycles
- Empirical roofline: 0.14 TFLOPS, 70.86 GB/s
- Roofline intensity: DRAM=2.02, VL1=2.19, L2C=6.59 FLOP/Byte
- Roofline usage: HBM=3.84%, VL1=0.44%, L2C=0.47%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=1.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=12.31%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
