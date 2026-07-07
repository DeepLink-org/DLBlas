# C500 Trace Profile Digest: baseline

## Kernel

- Name: `_Z19big_fuse_kernel_optPKDF16_PKfS2_S2_PfS3_PDF16_iiiiiffffi`
- Grid: `[512, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `3,524,486` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 49.44% / 34.28%
- GVM share / VLS duty / L2C duty: 2.88% / 0.00% / 0.00%
- VL1/L2C hit rate: 96.98% / 96.77%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 1.00%
- WIF P25/P50/P75/max: 37.00 / 110.00 / 183.00 / 208
- DPC balance: min/avg/max=13,857,664/14,664,905/14,782,576 cycles, imbalance=6.31%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 3,524,486 | cycles | lower |
| CycleTrace instructions | 91,176 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 49.44 | % | context |
| CycleTrace BSM share | 0.6581 | % | context |
| CycleTrace GVM share | 2.879 | % | lower |
| CycleTrace ARRIVE share | 7.118 | % | lower |
| CycleTrace LDU share | 0.0658 | % | context |
| CycleTrace NOP share | 13.13 | % | lower |
| Wave in flight mean | 108.4 | waves | higher |
| Wave in flight max | 208 | waves | higher |
| Wave in flight P25 | 37 | waves | context |
| Wave in flight P50 | 110 | waves | context |
| Wave in flight P75 | 183 | waves | context |
| GVM issue peak / 600 cycles | 13 | inst | lower |
| Registers/thread | 108 | regs | lower |
| Static shared | 1,024 | bytes | lower |
| mtreg occupancy | 21 | % | higher |
| shared memory occupancy | 1 | % | higher |
| digest occupancy bound | 1 | % | higher |
| AP MTE duty | 34.28 | % | higher |
| AP STE duty | 14.42 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 53.19 | inst/cycle | higher |
| Instruction throughput efficiency | 12.78 | % | higher |
| Instructions per AP | 314,735 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 78.7 | % | lower |
| ISU stall cycles | 2,032,198 | cycles | lower |
| VL1 hit rate | 96.98 | % | higher |
| L2C hit rate | 96.77 | % | higher |
| DNOC read requests | 52,467 | req | lower |
| DNOC write requests | 15,103 | req | lower |
| Global memory read bytes | 6,323,360 | bytes | lower |
| Global memory write bytes | 1,927,264 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 2,300 | waves | higher |
| Dispatched waves | 2,316 | waves | context |
| Average wave life | 76,493 | cycles | lower |
| Empirical achieved FLOPS | 2.52 | TFLOPS | higher |
| Empirical achieved bandwidth | 15.08 | GB/s | higher |
| Empirical roofline intensity | 167.1 | FLOP/Byte | context |
| Roofline HBM usage | 0.8182 | % | higher |
| Roofline VL1 usage | 3.329 | % | higher |
| Roofline L2C usage | 10.32 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 5.055 | FLOP/Byte | context |
| L2C-level roofline intensity | 5.3 | FLOP/Byte | context |
| DPC compute imbalance | 6.307 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 45,075 | 49.44% | Vector/Matrix Tensor Extension compute category |
| STE | 36,326 | 39.84% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 600 | 0.66% | Block shared-memory hardware category |
| GLOBAL | 2,625 | 2.88% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 6,490 | 7.12% | Synchronization/arrival category |
| LDU | 60 | 0.07% | Load data unit category |

- STE subevents by `name`: STE=21,280, S_NOP=11,976, Branch=3,070.
- GLOBAL subevents by `name`: GVM Load=2,600, GVM Store=25.
- ARRIVE subevents by `name`: Synchronization=6,490.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 34.28% / 14.42% / 0.00%
- Real IPC: 53.19
- VL1/L2C hit rate: 96.98% / 96.77%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=13,857,664/14,664,905/14,782,576 cycles, imbalance=6.31%
- Achieved/dispatched waves: 2,300 / 2,316; workgroups=579; avg wave life=76,493 cycles
- Empirical roofline: 2.52 TFLOPS, 15.08 GB/s
- Roofline intensity: DRAM=167.09, VL1=5.05, L2C=5.30 FLOP/Byte
- Roofline usage: HBM=0.82%, VL1=3.33%, L2C=10.32%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=21.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=13.13%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
