# C500 Trace Profile Digest: baseline

## Kernel

- Name: `norm_fn_kernel_opt(float const*, float const*, float*, int, int, int, float)`
- Grid: `[24, 13, 1]`
- Block: `[256, 1, 1]`
- Span: `2,039,819` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 36.97% / 16.00%
- GVM share / VLS duty / L2C duty: 5.17% / 0.00% / 0.03%
- VL1/L2C hit rate: 96.55% / 83.85%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 1.00%
- WIF P25/P50/P75/max: 16.00 / 54.00 / 105.00 / 156
- DPC balance: min/avg/max=223,188/233,362/237,448 cycles, imbalance=6.11%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 2,039,819 | cycles | lower |
| CycleTrace instructions | 2,321 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 36.97 | % | context |
| CycleTrace BSM share | 0.3878 | % | context |
| CycleTrace GVM share | 5.17 | % | lower |
| CycleTrace ARRIVE share | 12.67 | % | lower |
| CycleTrace LDU share | 0.6463 | % | context |
| CycleTrace NOP share | 18.05 | % | lower |
| Wave in flight mean | 62.08 | waves | higher |
| Wave in flight max | 156 | waves | higher |
| Wave in flight P25 | 16 | waves | context |
| Wave in flight P50 | 54 | waves | context |
| Wave in flight P75 | 105 | waves | context |
| GVM issue peak / 600 cycles | 12 | inst | lower |
| Registers/thread | 14 | regs | lower |
| Static shared | 1,024 | bytes | lower |
| mtreg occupancy | 2 | % | higher |
| shared memory occupancy | 1 | % | higher |
| digest occupancy bound | 1 | % | higher |
| AP MTE duty | 16 | % | higher |
| AP STE duty | 6.957 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 29.84 | inst/cycle | higher |
| Instruction throughput efficiency | 7.172 | % | higher |
| Instructions per AP | 8,950 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 70.97 | % | lower |
| ISU stall cycles | 106,716 | cycles | lower |
| VL1 hit rate | 96.55 | % | higher |
| L2C hit rate | 83.85 | % | higher |
| DNOC read requests | 12,610 | req | lower |
| DNOC write requests | 5,993 | req | lower |
| Global memory read bytes | 1,178,656 | bytes | lower |
| Global memory write bytes | 763,008 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 1,600 | waves | higher |
| Dispatched waves | 1,616 | waves | context |
| Average wave life | 5,381 | cycles | lower |
| Empirical achieved FLOPS | 1.137 | TFLOPS | higher |
| Empirical achieved bandwidth | 70.02 | GB/s | higher |
| Empirical roofline intensity | 16.24 | FLOP/Byte | context |
| Roofline HBM usage | 3.799 | % | higher |
| Roofline VL1 usage | 3.093 | % | higher |
| Roofline L2C usage | 9.905 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 2.455 | FLOP/Byte | context |
| L2C-level roofline intensity | 2.492 | FLOP/Byte | context |
| DPC compute imbalance | 6.111 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 858 | 36.97% | Vector/Matrix Tensor Extension compute category |
| STE | 1,025 | 44.16% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 9 | 0.39% | Block shared-memory hardware category |
| GLOBAL | 120 | 5.17% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 294 | 12.67% | Synchronization/arrival category |
| LDU | 15 | 0.65% | Load data unit category |

- STE subevents by `name`: STE=432, S_NOP=419, Branch=174.
- GLOBAL subevents by `name`: GVM Load=120, GVM Store=0.
- ARRIVE subevents by `name`: Synchronization=294.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 16.00% / 6.96% / 0.00%
- Real IPC: 29.84
- VL1/L2C hit rate: 96.55% / 83.85%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=223,188/233,362/237,448 cycles, imbalance=6.11%
- Achieved/dispatched waves: 1,600 / 1,616; workgroups=404; avg wave life=5,381 cycles
- Empirical roofline: 1.14 TFLOPS, 70.02 GB/s
- Roofline intensity: DRAM=16.24, VL1=2.45, L2C=2.49 FLOP/Byte
- Roofline usage: HBM=3.80%, VL1=3.09%, L2C=9.90%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=2.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=18.05%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
