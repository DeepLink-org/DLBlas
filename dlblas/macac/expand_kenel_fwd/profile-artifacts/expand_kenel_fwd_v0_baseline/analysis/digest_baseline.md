# C500 Trace Profile Digest: baseline

## Kernel

- Name: `expand_kenel_fwd_kernel_opt(float const*, float*, int, int, int, int)`
- Grid: `[1024, 1, 1]`
- Block: `[320, 1, 1]`
- Span: `23,801` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 37.37% / 8.32%
- GVM share / VLS duty / L2C duty: 6.67% / 0.00% / 0.09%
- VL1/L2C hit rate: 77.50% / 0.72%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 2.00%
- WIF P25/P50/P75/max: 160.00 / 320.00 / 382.00 / 392
- DPC balance: min/avg/max=66,080/70,980/71,680 cycles, imbalance=7.89%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 23,801 | cycles | lower |
| CycleTrace instructions | 974 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 37.37 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 6.673 | % | lower |
| CycleTrace ARRIVE share | 6.673 | % | lower |
| CycleTrace LDU share | 4.004 | % | context |
| CycleTrace NOP share | 7.905 | % | lower |
| Wave in flight mean | 268.3 | waves | higher |
| Wave in flight max | 392 | waves | higher |
| Wave in flight P25 | 160 | waves | context |
| Wave in flight P50 | 320 | waves | context |
| Wave in flight P75 | 382 | waves | context |
| GVM issue peak / 600 cycles | 6 | inst | lower |
| Registers/thread | 12 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 2 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 2 | % | higher |
| AP MTE duty | 8.325 | % | higher |
| AP STE duty | 8.02 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 20.58 | inst/cycle | higher |
| Instruction throughput efficiency | 4.947 | % | higher |
| Instructions per AP | 3,415 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 54.68 | % | lower |
| ISU stall cycles | 716,638 | cycles | lower |
| VL1 hit rate | 77.5 | % | higher |
| L2C hit rate | 0.7198 | % | higher |
| DNOC read requests | 41,000 | req | lower |
| DNOC write requests | 163,861 | req | lower |
| Global memory read bytes | 5,245,152 | bytes | lower |
| Global memory write bytes | 20,972,160 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 5,075 | waves | higher |
| Dispatched waves | 5,120 | waves | context |
| Average wave life | 2,689 | cycles | lower |
| Empirical achieved FLOPS | 0.5929 | TFLOPS | higher |
| Empirical achieved bandwidth | 1,709 | GB/s | higher |
| Empirical roofline intensity | 0.3469 | FLOP/Byte | context |
| Roofline HBM usage | 92.74 | % | higher |
| Roofline VL1 usage | 2.828 | % | higher |
| Roofline L2C usage | 7.353 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 1.4 | FLOP/Byte | context |
| L2C-level roofline intensity | 1.75 | FLOP/Byte | context |
| DPC compute imbalance | 7.889 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 364 | 37.37% | Vector/Matrix Tensor Extension compute category |
| STE | 441 | 45.28% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 65 | 6.67% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 65 | 6.67% | Synchronization/arrival category |
| LDU | 39 | 4.00% | Load data unit category |

- STE subevents by `name`: STE=351, S_NOP=77, Branch=13.
- GLOBAL subevents by `name`: GVM Load=13, GVM Store=52.
- ARRIVE subevents by `name`: Synchronization=65.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 8.32% / 8.02% / 0.00%
- Real IPC: 20.58
- VL1/L2C hit rate: 77.50% / 0.72%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=66,080/70,980/71,680 cycles, imbalance=7.89%
- Achieved/dispatched waves: 5,075 / 5,120; workgroups=1,024; avg wave life=2,689 cycles
- Empirical roofline: 0.59 TFLOPS, 1709.33 GB/s
- Roofline intensity: DRAM=0.35, VL1=1.40, L2C=1.75 FLOP/Byte
- Roofline usage: HBM=92.74%, VL1=2.83%, L2C=7.35%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=2.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
