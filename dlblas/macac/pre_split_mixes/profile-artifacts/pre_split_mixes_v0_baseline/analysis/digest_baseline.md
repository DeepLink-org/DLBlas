# C500 Trace Profile Digest: baseline

## Kernel

- Name: `pre_split_mixes_kernel_opt(float const*, float const*, float const*, float*, float*, float*, int, int, int, float, float)`
- Grid: `[2, 1, 1]`
- Block: `[512, 1, 1]`
- Span: `26,163,126` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 59.56% / 20.34%
- GVM share / VLS duty / L2C duty: 2.64% / 0.00% / 0.00%
- VL1/L2C hit rate: 25.15% / 78.84%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 4.00%
- WIF P25/P50/P75/max: 2.00 / 4.00 / 6.00 / 12
- DPC balance: min/avg/max=1,937,888/2,072,015/2,104,496 cycles, imbalance=8.04%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 26,163,126 | cycles | lower |
| CycleTrace instructions | 8,190 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 59.56 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 2.637 | % | lower |
| CycleTrace ARRIVE share | 3.517 | % | lower |
| CycleTrace LDU share | 2.637 | % | context |
| CycleTrace NOP share | 7.253 | % | lower |
| Wave in flight mean | 4.008 | waves | higher |
| Wave in flight max | 12 | waves | higher |
| Wave in flight P25 | 2 | waves | context |
| Wave in flight P50 | 4 | waves | context |
| Wave in flight P75 | 6 | waves | context |
| GVM issue peak / 600 cycles | 7 | inst | lower |
| Registers/thread | 22 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 4 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 4 | % | higher |
| AP MTE duty | 20.34 | % | higher |
| AP STE duty | 6.215 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 25.07 | inst/cycle | higher |
| Instruction throughput efficiency | 6.026 | % | higher |
| Instructions per AP | 33,172 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 76.14 | % | lower |
| ISU stall cycles | 3,238,433 | cycles | lower |
| VL1 hit rate | 25.15 | % | higher |
| L2C hit rate | 78.84 | % | higher |
| DNOC read requests | 443,001 | req | lower |
| DNOC write requests | 394,673 | req | lower |
| Global memory read bytes | 52,919,424 | bytes | lower |
| Global memory write bytes | 49,788,864 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 8,292 | waves | higher |
| Dispatched waves | 8,376 | waves | context |
| Average wave life | 2,597 | cycles | lower |
| Empirical achieved FLOPS | 1.36 | TFLOPS | higher |
| Empirical achieved bandwidth | 839.6 | GB/s | higher |
| Empirical roofline intensity | 1.62 | FLOP/Byte | context |
| Roofline HBM usage | 45.55 | % | higher |
| Roofline VL1 usage | 1.345 | % | higher |
| Roofline L2C usage | 52.32 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 6.751 | FLOP/Byte | context |
| L2C-level roofline intensity | 0.5643 | FLOP/Byte | context |
| DPC compute imbalance | 8.041 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 4,878 | 59.56% | Vector/Matrix Tensor Extension compute category |
| STE | 2,592 | 31.65% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 216 | 2.64% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 288 | 3.52% | Synchronization/arrival category |
| LDU | 216 | 2.64% | Load data unit category |

- STE subevents by `name`: STE=1,764, S_NOP=594, Branch=234.
- GLOBAL subevents by `name`: GVM Load=108, GVM Store=108.
- ARRIVE subevents by `name`: Synchronization=288.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 20.34% / 6.22% / 0.00%
- Real IPC: 25.07
- VL1/L2C hit rate: 25.15% / 78.84%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=1,937,888/2,072,015/2,104,496 cycles, imbalance=8.04%
- Achieved/dispatched waves: 8,292 / 8,376; workgroups=1,084; avg wave life=2,597 cycles
- Empirical roofline: 1.36 TFLOPS, 839.62 GB/s
- Roofline intensity: DRAM=1.62, VL1=6.75, L2C=0.56 FLOP/Byte
- Roofline usage: HBM=45.55%, VL1=1.35%, L2C=52.32%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=4.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
