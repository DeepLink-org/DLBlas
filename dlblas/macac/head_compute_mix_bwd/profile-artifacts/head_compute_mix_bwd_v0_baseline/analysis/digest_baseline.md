# C500 Trace Profile Digest: baseline

## Kernel

- Name: `head_compute_mix_bwd_kernel_opt(float const*, float const*, float const*, float const*, float*, float*, float*, int, int, int)`
- Grid: `[16, 1, 1]`
- Block: `[512, 1, 1]`
- Span: `1,637,173` cycles

## Bound Classification

- Mode: `coarse`
- Type: `occupancy`
- Primary: `occupancy`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 29.07% / 12.69%
- GVM share / VLS duty / L2C duty: 0.40% / 0.00% / 0.00%
- VL1/L2C hit rate: 92.06% / 13.88%
- Shared-memory efficiency / conflict cycles: 99.45% / 0.01115
- Effective occupancy bound: 2.00%
- WIF P25/P50/P75/max: 1.75 / 3.00 / 7.00 / 16
- DPC balance: min/avg/max=34,832/36,962/39,704 cycles, imbalance=13.18%

Rationale:

- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 1,637,173 | cycles | lower |
| CycleTrace instructions | 2,023 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 29.07 | % | context |
| CycleTrace BSM share | 1.631 | % | context |
| CycleTrace GVM share | 0.3955 | % | lower |
| CycleTrace ARRIVE share | 14.98 | % | lower |
| CycleTrace LDU share | 0.692 | % | context |
| CycleTrace NOP share | 21.75 | % | lower |
| Wave in flight mean | 4.667 | waves | higher |
| Wave in flight max | 16 | waves | higher |
| Wave in flight P25 | 1.75 | waves | context |
| Wave in flight P50 | 3 | waves | context |
| Wave in flight P75 | 7 | waves | context |
| GVM issue peak / 600 cycles | 6 | inst | lower |
| Registers/thread | 15 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 2 | % | higher |
| shared memory occupancy | 15 | % | higher |
| digest occupancy bound | 2 | % | higher |
| AP MTE duty | 12.69 | % | higher |
| AP STE duty | 6.069 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 8.743 | inst/cycle | higher |
| Instruction throughput efficiency | 2.102 | % | higher |
| Instructions per AP | 1,318 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 52.07 | % | lower |
| ISU stall cycles | 7,908 | cycles | lower |
| VL1 hit rate | 92.06 | % | higher |
| L2C hit rate | 13.88 | % | higher |
| DNOC read requests | 5,001 | req | lower |
| DNOC write requests | 1,379 | req | lower |
| Global memory read bytes | 351,840 | bytes | lower |
| Global memory write bytes | 167,008 | bytes | lower |
| Shared memory efficiency | 99.45 | % | higher |
| Avg conflict cycles/inst | 0.01115 | cycles | lower |
| Achieved waves | 280 | waves | higher |
| Dispatched waves | 284 | waves | context |
| Average wave life | 3,663 | cycles | lower |
| Empirical achieved FLOPS | 0.2763 | TFLOPS | higher |
| Empirical achieved bandwidth | 37.24 | GB/s | higher |
| Empirical roofline intensity | 7.419 | FLOP/Byte | context |
| Roofline HBM usage | 2.02 | % | higher |
| Roofline VL1 usage | 0.0782 | % | higher |
| Roofline L2C usage | 0.1276 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 23.59 | FLOP/Byte | context |
| L2C-level roofline intensity | 46.99 | FLOP/Byte | context |
| DPC compute imbalance | 13.18 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 588 | 29.07% | Vector/Matrix Tensor Extension compute category |
| STE | 1,077 | 53.24% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 33 | 1.63% | Block shared-memory hardware category |
| GLOBAL | 8 | 0.40% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 303 | 14.98% | Synchronization/arrival category |
| LDU | 14 | 0.69% | Load data unit category |

- STE subevents by `name`: STE=402, S_NOP=440, Branch=235.
- GLOBAL subevents by `name`: GVM Load=6, GVM Store=2.
- ARRIVE subevents by `name`: Synchronization=303.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 12.69% / 6.07% / 0.00%
- Real IPC: 8.74
- VL1/L2C hit rate: 92.06% / 13.88%
- Shared memory efficiency: 99.45%
- Avg conflict cycles/inst: 0.01115
- DPC compute balance: min/avg/max=34,832/36,962/39,704 cycles, imbalance=13.18%
- Achieved/dispatched waves: 280 / 284; workgroups=55; avg wave life=3,663 cycles
- Empirical roofline: 0.28 TFLOPS, 37.24 GB/s
- Roofline intensity: DRAM=7.42, VL1=23.59, L2C=46.99 FLOP/Byte
- Roofline usage: HBM=2.02%, VL1=0.08%, L2C=0.13%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=2.00%, smem=15.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=21.75%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
