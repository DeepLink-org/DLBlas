# C500 Trace Profile Digest: baseline

## Kernel

- Name: `apply_mix_kernel_opt(unsigned short const*, float const*, unsigned short*, int, int, int, int)`
- Grid: `[2048, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `29,689,532` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 51.97% / 41.75%
- GVM share / VLS duty / L2C duty: 8.50% / 0.00% / 0.03%
- VL1/L2C hit rate: 96.53% / 44.82%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 5.00%
- WIF P25/P50/P75/max: 253.00 / 320.00 / 345.00 / 416
- DPC balance: min/avg/max=543,824/579,608/584,720 cycles, imbalance=7.06%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 29,689,532 | cycles | lower |
| CycleTrace instructions | 6,032 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 51.97 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 8.505 | % | lower |
| CycleTrace ARRIVE share | 9.765 | % | lower |
| CycleTrace LDU share | 1.873 | % | context |
| CycleTrace NOP share | 6.83 | % | lower |
| Wave in flight mean | 282.3 | waves | higher |
| Wave in flight max | 416 | waves | higher |
| Wave in flight P25 | 253 | waves | context |
| Wave in flight P50 | 320 | waves | context |
| Wave in flight P75 | 345 | waves | context |
| GVM issue peak / 600 cycles | 34 | inst | lower |
| Registers/thread | 26 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 5 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 5 | % | higher |
| AP MTE duty | 41.75 | % | higher |
| AP STE duty | 17.57 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 52.48 | inst/cycle | higher |
| Instruction throughput efficiency | 12.62 | % | higher |
| Instructions per AP | 20,544 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 93.18 | % | lower |
| ISU stall cycles | 1,196,213 | cycles | lower |
| VL1 hit rate | 96.53 | % | higher |
| L2C hit rate | 44.82 | % | higher |
| DNOC read requests | 165,301 | req | lower |
| DNOC write requests | 41,247 | req | lower |
| Global memory read bytes | 21,064,960 | bytes | lower |
| Global memory write bytes | 5,276,352 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 8,144 | waves | higher |
| Dispatched waves | 8,224 | waves | context |
| Average wave life | 2,033 | cycles | lower |
| Empirical achieved FLOPS | 2.261 | TFLOPS | higher |
| Empirical achieved bandwidth | 727.9 | GB/s | higher |
| Empirical roofline intensity | 3.107 | FLOP/Byte | context |
| Roofline HBM usage | 39.49 | % | higher |
| Roofline VL1 usage | 8.632 | % | higher |
| Roofline L2C usage | 24.93 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 1.749 | FLOP/Byte | context |
| L2C-level roofline intensity | 1.968 | FLOP/Byte | context |
| DPC compute imbalance | 7.056 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 3,135 | 51.97% | Vector/Matrix Tensor Extension compute category |
| STE | 1,682 | 27.88% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 513 | 8.50% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 589 | 9.76% | Synchronization/arrival category |
| LDU | 113 | 1.87% | Load data unit category |

- STE subevents by `name`: STE=1,175, S_NOP=412, Branch=95.
- GLOBAL subevents by `name`: GVM Load=456, GVM Store=57.
- ARRIVE subevents by `name`: Synchronization=589.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 41.75% / 17.57% / 0.00%
- Real IPC: 52.48
- VL1/L2C hit rate: 96.53% / 44.82%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=543,824/579,608/584,720 cycles, imbalance=7.06%
- Achieved/dispatched waves: 8,144 / 8,224; workgroups=2,056; avg wave life=2,033 cycles
- Empirical roofline: 2.26 TFLOPS, 727.91 GB/s
- Roofline intensity: DRAM=3.11, VL1=1.75, L2C=1.97 FLOP/Byte
- Roofline usage: HBM=39.49%, VL1=8.63%, L2C=24.93%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=5.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
