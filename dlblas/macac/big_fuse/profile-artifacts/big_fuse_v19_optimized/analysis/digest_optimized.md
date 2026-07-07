# C500 Trace Profile Digest: optimized

## Kernel

- Name: `_Z19big_fuse_kernel_optPKDF16_PKfS2_S2_PfS3_PDF16_iiiiiffffi`
- Grid: `[512, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `3,323,867` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 49.58% / 55.13%
- GVM share / VLS duty / L2C duty: 3.25% / 0.00% / 0.22%
- VL1/L2C hit rate: 96.98% / 96.71%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 1.00%
- WIF P25/P50/P75/max: 40.00 / 112.00 / 184.00 / 256
- DPC balance: min/avg/max=12,696,188/13,636,040/13,772,656 cycles, imbalance=7.89%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 3,323,867 | cycles | lower |
| CycleTrace instructions | 80,724 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 49.58 | % | context |
| CycleTrace BSM share | 0.9663 | % | context |
| CycleTrace GVM share | 3.252 | % | lower |
| CycleTrace ARRIVE share | 4.249 | % | lower |
| CycleTrace LDU share | 0.0743 | % | context |
| CycleTrace NOP share | 11.75 | % | lower |
| Wave in flight mean | 114.6 | waves | higher |
| Wave in flight max | 256 | waves | higher |
| Wave in flight P25 | 40 | waves | context |
| Wave in flight P50 | 112 | waves | context |
| Wave in flight P75 | 184 | waves | context |
| GVM issue peak / 600 cycles | 12 | inst | lower |
| Registers/thread | 50 | regs | lower |
| Static shared | 1,024 | bytes | lower |
| mtreg occupancy | 9 | % | higher |
| shared memory occupancy | 1 | % | higher |
| digest occupancy bound | 1 | % | higher |
| AP MTE duty | 55.13 | % | higher |
| AP STE duty | 28.9 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 87.46 | inst/cycle | higher |
| Instruction throughput efficiency | 21.02 | % | higher |
| Instructions per AP | 285,188 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 72.03 | % | lower |
| ISU stall cycles | 2,220,568 | cycles | lower |
| VL1 hit rate | 96.98 | % | higher |
| L2C hit rate | 96.71 | % | higher |
| DNOC read requests | 54,302 | req | lower |
| DNOC write requests | 15,112 | req | lower |
| Global memory read bytes | 6,544,032 | bytes | lower |
| Global memory write bytes | 1,927,232 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 2,292 | waves | higher |
| Dispatched waves | 2,316 | waves | context |
| Average wave life | 79,047 | cycles | lower |
| Empirical achieved FLOPS | 4.156 | TFLOPS | higher |
| Empirical achieved bandwidth | 28.1 | GB/s | higher |
| Empirical roofline intensity | 147.9 | FLOP/Byte | context |
| Roofline HBM usage | 1.525 | % | higher |
| Roofline VL1 usage | 6.042 | % | higher |
| Roofline L2C usage | 18.69 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 4.593 | FLOP/Byte | context |
| L2C-level roofline intensity | 4.826 | FLOP/Byte | context |
| DPC compute imbalance | 7.894 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 40,025 | 49.58% | Vector/Matrix Tensor Extension compute category |
| STE | 33,804 | 41.88% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 780 | 0.97% | Block shared-memory hardware category |
| GLOBAL | 2,625 | 3.25% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 3,430 | 4.25% | Synchronization/arrival category |
| LDU | 60 | 0.07% | Load data unit category |

- STE subevents by `name`: STE=23,238, S_NOP=9,486, Branch=1,080.
- GLOBAL subevents by `name`: GVM Load=2,600, GVM Store=25.
- ARRIVE subevents by `name`: Synchronization=3,430.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 55.13% / 28.90% / 0.00%
- Real IPC: 87.46
- VL1/L2C hit rate: 96.98% / 96.71%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=12,696,188/13,636,040/13,772,656 cycles, imbalance=7.89%
- Achieved/dispatched waves: 2,292 / 2,316; workgroups=579; avg wave life=79,047 cycles
- Empirical roofline: 4.16 TFLOPS, 28.10 GB/s
- Roofline intensity: DRAM=147.88, VL1=4.59, L2C=4.83 FLOP/Byte
- Roofline usage: HBM=1.52%, VL1=6.04%, L2C=18.69%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=9.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=11.75%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
