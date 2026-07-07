# C500 Trace Profile Digest: baseline

## Kernel

- Name: `mhc_post_kernel_opt(__maca_bfloat16 const*, __maca_bfloat16 const*, float const*, float const*, __maca_bfloat16*, int, int, int, int)`
- Grid: `[832, 1, 1]`
- Block: `[512, 1, 1]`
- Span: `1,068,811,235` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 62.60% / 69.15%
- GVM share / VLS duty / L2C duty: 7.73% / 0.00% / 0.00%
- VL1/L2C hit rate: 97.73% / 0.58%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 8.00%
- WIF P25/P50/P75/max: 181.00 / 318.00 / 357.00 / 416
- DPC balance: min/avg/max=5,541,664/5,945,722/6,005,168 cycles, imbalance=7.80%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 1,068,811,235 | cycles | lower |
| CycleTrace instructions | 45,969 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 62.6 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 7.734 | % | lower |
| CycleTrace ARRIVE share | 5.295 | % | lower |
| CycleTrace LDU share | 1.199 | % | context |
| CycleTrace NOP share | 4.69 | % | lower |
| Wave in flight mean | 266.7 | waves | higher |
| Wave in flight max | 416 | waves | higher |
| Wave in flight P25 | 181 | waves | context |
| Wave in flight P50 | 318 | waves | context |
| Wave in flight P75 | 357 | waves | context |
| GVM issue peak / 600 cycles | 34 | inst | lower |
| Registers/thread | 42 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 8 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 8 | % | higher |
| AP MTE duty | 69.15 | % | higher |
| AP STE duty | 19.06 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 104.1 | inst/cycle | higher |
| Instruction throughput efficiency | 25.03 | % | higher |
| Instructions per AP | 174,051 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 64.67 | % | lower |
| ISU stall cycles | 7,377,417 | cycles | lower |
| VL1 hit rate | 97.73 | % | higher |
| L2C hit rate | 0.5849 | % | higher |
| DNOC read requests | 834,668 | req | lower |
| DNOC write requests | 660,511 | req | lower |
| Global memory read bytes | 105,854,912 | bytes | lower |
| Global memory write bytes | 84,542,400 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 6,896 | waves | higher |
| Dispatched waves | 6,964 | waves | context |
| Average wave life | 16,043 | cycles | lower |
| Empirical achieved FLOPS | 5.997 | TFLOPS | higher |
| Empirical achieved bandwidth | 1,232 | GB/s | higher |
| Empirical roofline intensity | 4.867 | FLOP/Byte | context |
| Roofline HBM usage | 66.86 | % | higher |
| Roofline VL1 usage | 16.16 | % | higher |
| Roofline L2C usage | 14.59 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 2.477 | FLOP/Byte | context |
| L2C-level roofline intensity | 8.922 | FLOP/Byte | context |
| DPC compute imbalance | 7.796 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 28,776 | 62.60% | Vector/Matrix Tensor Extension compute category |
| STE | 10,653 | 23.17% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 3,555 | 7.73% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 2,434 | 5.29% | Synchronization/arrival category |
| LDU | 551 | 1.20% | Load data unit category |

- STE subevents by `name`: STE=7,912, S_NOP=2,156, Branch=585.
- GLOBAL subevents by `name`: GVM Load=1,975, GVM Store=1,580.
- ARRIVE subevents by `name`: Synchronization=2,434.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 69.15% / 19.06% / 0.00%
- Real IPC: 104.14
- VL1/L2C hit rate: 97.73% / 0.58%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=5,541,664/5,945,722/6,005,168 cycles, imbalance=7.80%
- Achieved/dispatched waves: 6,896 / 6,964; workgroups=909; avg wave life=16,043 cycles
- Empirical roofline: 6.00 TFLOPS, 1232.36 GB/s
- Roofline intensity: DRAM=4.87, VL1=2.48, L2C=8.92 FLOP/Byte
- Roofline usage: HBM=66.86%, VL1=16.16%, L2C=14.59%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=8.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
