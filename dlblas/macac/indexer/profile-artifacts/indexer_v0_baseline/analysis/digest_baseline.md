# C500 Trace Profile Digest: baseline

## Kernel

- Name: `indexer_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, int*, int, int, int, int, int, int, int, int)`
- Grid: `[128, 1, 1]`
- Block: `[64, 1, 1]`
- Span: `1,798,471` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 47.10% / 3.40%
- GVM share / VLS duty / L2C duty: 11.03% / 0.00% / 0.00%
- VL1/L2C hit rate: 84.41% / 98.46%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 5.00%
- WIF P25/P50/P75/max: 4.00 / 8.00 / 14.00 / 24
- DPC balance: min/avg/max=881,852/1,039,926/1,148,312 cycles, imbalance=25.62%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 1,798,471 | cycles | lower |
| CycleTrace instructions | 4,786 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 47.1 | % | context |
| CycleTrace BSM share | 1.504 | % | context |
| CycleTrace GVM share | 11.03 | % | lower |
| CycleTrace ARRIVE share | 11.64 | % | lower |
| CycleTrace LDU share | 0.0836 | % | context |
| CycleTrace NOP share | 9.047 | % | lower |
| Wave in flight mean | 9.333 | waves | higher |
| Wave in flight max | 24 | waves | higher |
| Wave in flight P25 | 4 | waves | context |
| Wave in flight P50 | 8 | waves | context |
| Wave in flight P75 | 14 | waves | context |
| GVM issue peak / 600 cycles | 16 | inst | lower |
| Registers/thread | 30 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 5 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 5 | % | higher |
| AP MTE duty | 3.399 | % | higher |
| AP STE duty | 1.085 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 5.28 | inst/cycle | higher |
| Instruction throughput efficiency | 1.269 | % | higher |
| Instructions per AP | 5,908 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 45.92 | % | lower |
| ISU stall cycles | 635,261 | cycles | lower |
| VL1 hit rate | 84.41 | % | higher |
| L2C hit rate | 98.46 | % | higher |
| DNOC read requests | 6,979 | req | lower |
| DNOC write requests | 2,831 | req | lower |
| Global memory read bytes | 579,360 | bytes | lower |
| Global memory write bytes | 357,952 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 387 | waves | higher |
| Dispatched waves | 392 | waves | context |
| Average wave life | 10,723 | cycles | lower |
| Empirical achieved FLOPS | 0.2236 | TFLOPS | higher |
| Empirical achieved bandwidth | 9.061 | GB/s | higher |
| Empirical roofline intensity | 24.67 | FLOP/Byte | context |
| Roofline HBM usage | 0.4916 | % | higher |
| Roofline VL1 usage | 1.081 | % | higher |
| Roofline L2C usage | 17.32 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 1.381 | FLOP/Byte | context |
| L2C-level roofline intensity | 0.2801 | FLOP/Byte | context |
| DPC compute imbalance | 25.62 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 2,254 | 47.10% | Vector/Matrix Tensor Extension compute category |
| STE | 1,371 | 28.65% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 72 | 1.50% | Block shared-memory hardware category |
| GLOBAL | 528 | 11.03% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 557 | 11.64% | Synchronization/arrival category |
| LDU | 4 | 0.08% | Load data unit category |

- STE subevents by `name`: STE=765, S_NOP=433, Branch=173.
- GLOBAL subevents by `name`: GVM Load=512, GVM Store=16.
- ARRIVE subevents by `name`: Synchronization=557.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 3.40% / 1.09% / 0.00%
- Real IPC: 5.28
- VL1/L2C hit rate: 84.41% / 98.46%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=881,852/1,039,926/1,148,312 cycles, imbalance=25.62%
- Achieved/dispatched waves: 387 / 392; workgroups=194; avg wave life=10,723 cycles
- Empirical roofline: 0.22 TFLOPS, 9.06 GB/s
- Roofline intensity: DRAM=24.67, VL1=1.38, L2C=0.28 FLOP/Byte
- Roofline usage: HBM=0.49%, VL1=1.08%, L2C=17.32%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=5.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
