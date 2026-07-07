# C500 Trace Profile Digest: final

## Kernel

- Name: `norm_fn_kernel_opt(float const*, float const*, float*, int, int, int, float)`
- Grid: `[24, 13, 1]`
- Block: `[256, 1, 1]`
- Span: `2,048,215` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 39.80% / 14.56%
- GVM share / VLS duty / L2C duty: 2.55% / 0.00% / 0.03%
- VL1/L2C hit rate: 86.77% / 83.87%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 3.00%
- WIF P25/P50/P75/max: 17.00 / 54.00 / 105.00 / 156
- DPC balance: min/avg/max=138,144/151,246/155,484 cycles, imbalance=11.46%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 2,048,215 | cycles | lower |
| CycleTrace instructions | 1,176 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 39.8 | % | context |
| CycleTrace BSM share | 0.2551 | % | context |
| CycleTrace GVM share | 2.551 | % | lower |
| CycleTrace ARRIVE share | 10.46 | % | lower |
| CycleTrace LDU share | 1.276 | % | context |
| CycleTrace NOP share | 17.35 | % | lower |
| Wave in flight mean | 62.47 | waves | higher |
| Wave in flight max | 156 | waves | higher |
| Wave in flight P25 | 17 | waves | context |
| Wave in flight P50 | 54 | waves | context |
| Wave in flight P75 | 105 | waves | context |
| GVM issue peak / 600 cycles | 8 | inst | lower |
| Registers/thread | 22 | regs | lower |
| Static shared | 2,048 | bytes | lower |
| mtreg occupancy | 4 | % | higher |
| shared memory occupancy | 3 | % | higher |
| digest occupancy bound | 3 | % | higher |
| AP MTE duty | 14.56 | % | higher |
| AP STE duty | 6.229 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 24.09 | inst/cycle | higher |
| Instruction throughput efficiency | 5.79 | % | higher |
| Instructions per AP | 5,101 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 88.32 | % | lower |
| ISU stall cycles | 221,008 | cycles | lower |
| VL1 hit rate | 86.77 | % | higher |
| L2C hit rate | 83.87 | % | higher |
| DNOC read requests | 12,171 | req | lower |
| DNOC write requests | 6,002 | req | lower |
| Global memory read bytes | 1,157,248 | bytes | lower |
| Global memory write bytes | 763,008 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 1,600 | waves | higher |
| Dispatched waves | 1,616 | waves | context |
| Average wave life | 3,682 | cycles | lower |
| Empirical achieved FLOPS | 1.035 | TFLOPS | higher |
| Empirical achieved bandwidth | 98.09 | GB/s | higher |
| Empirical roofline intensity | 10.55 | FLOP/Byte | context |
| Roofline HBM usage | 5.322 | % | higher |
| Roofline VL1 usage | 1.144 | % | higher |
| Roofline L2C usage | 14.03 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 6.041 | FLOP/Byte | context |
| L2C-level roofline intensity | 1.601 | FLOP/Byte | context |
| DPC compute imbalance | 11.46 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 468 | 39.80% | Vector/Matrix Tensor Extension compute category |
| STE | 537 | 45.66% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 3 | 0.26% | Block shared-memory hardware category |
| GLOBAL | 30 | 2.55% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 123 | 10.46% | Synchronization/arrival category |
| LDU | 15 | 1.28% | Load data unit category |

- STE subevents by `name`: STE=255, S_NOP=204, Branch=78.
- GLOBAL subevents by `name`: GVM Load=30, GVM Store=0.
- ARRIVE subevents by `name`: Synchronization=123.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 14.56% / 6.23% / 0.00%
- Real IPC: 24.09
- VL1/L2C hit rate: 86.77% / 83.87%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=138,144/151,246/155,484 cycles, imbalance=11.46%
- Achieved/dispatched waves: 1,600 / 1,616; workgroups=404; avg wave life=3,682 cycles
- Empirical roofline: 1.04 TFLOPS, 98.09 GB/s
- Roofline intensity: DRAM=10.55, VL1=6.04, L2C=1.60 FLOP/Byte
- Roofline usage: HBM=5.32%, VL1=1.14%, L2C=14.03%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=4.00%, smem=3.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=17.35%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
