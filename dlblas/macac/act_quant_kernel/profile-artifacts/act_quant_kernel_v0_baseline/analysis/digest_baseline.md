# C500 Trace Profile Digest: baseline

## Kernel

- Name: `_Z20act_quant_kernel_optPKDF16_PDF16_Pfiiiff`
- Grid: `[7, 1, 1]`
- Block: `[512, 1, 1]`
- Span: `1,700,848` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 37.36% / 12.80%
- GVM share / VLS duty / L2C duty: 1.10% / 0.00% / 0.00%
- VL1/L2C hit rate: 96.47% / 25.52%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 3.00%
- WIF P25/P50/P75/max: 1.75 / 3.00 / 5.00 / 8
- DPC balance: min/avg/max=2,868/8,038/9,024 cycles, imbalance=76.58%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 1,700,848 | cycles | lower |
| CycleTrace instructions | 728 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 37.36 | % | context |
| CycleTrace BSM share | 0.9615 | % | context |
| CycleTrace GVM share | 1.099 | % | lower |
| CycleTrace ARRIVE share | 11.26 | % | lower |
| CycleTrace LDU share | 1.648 | % | context |
| CycleTrace NOP share | 18.41 | % | lower |
| Wave in flight mean | 3.333 | waves | higher |
| Wave in flight max | 8 | waves | higher |
| Wave in flight P25 | 1.75 | waves | context |
| Wave in flight P50 | 3 | waves | context |
| Wave in flight P75 | 5 | waves | context |
| GVM issue peak / 600 cycles | 4 | inst | lower |
| Registers/thread | 24 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 4 | % | higher |
| shared memory occupancy | 3 | % | higher |
| digest occupancy bound | 3 | % | higher |
| AP MTE duty | 12.8 | % | higher |
| AP STE duty | 5.399 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 3.108 | inst/cycle | higher |
| Instruction throughput efficiency | 0.7472 | % | higher |
| Instructions per AP | 228.7 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 68.39 | % | lower |
| ISU stall cycles | 3,461 | cycles | lower |
| VL1 hit rate | 96.47 | % | higher |
| L2C hit rate | 25.52 | % | higher |
| DNOC read requests | 1,087 | req | lower |
| DNOC write requests | 274 | req | lower |
| Global memory read bytes | 72,832 | bytes | lower |
| Global memory write bytes | 30,560 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 88 | waves | higher |
| Dispatched waves | 88 | waves | context |
| Average wave life | 2,211 | cycles | lower |
| Empirical achieved FLOPS | 0.1183 | TFLOPS | higher |
| Empirical achieved bandwidth | 15.2 | GB/s | higher |
| Empirical roofline intensity | 7.783 | FLOP/Byte | context |
| Roofline HBM usage | 0.8247 | % | higher |
| Roofline VL1 usage | 0.0616 | % | higher |
| Roofline L2C usage | 0.0457 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 12.82 | FLOP/Byte | context |
| L2C-level roofline intensity | 56.13 | FLOP/Byte | context |
| DPC compute imbalance | 76.58 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 272 | 37.36% | Vector/Matrix Tensor Extension compute category |
| STE | 347 | 47.66% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 7 | 0.96% | Block shared-memory hardware category |
| GLOBAL | 8 | 1.10% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 82 | 11.26% | Synchronization/arrival category |
| LDU | 12 | 1.65% | Load data unit category |

- STE subevents by `name`: STE=158, S_NOP=134, Branch=55.
- GLOBAL subevents by `name`: GVM Load=4, GVM Store=4.
- ARRIVE subevents by `name`: Synchronization=82.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 12.80% / 5.40% / 0.00%
- Real IPC: 3.11
- VL1/L2C hit rate: 96.47% / 25.52%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=2,868/8,038/9,024 cycles, imbalance=76.58%
- Achieved/dispatched waves: 88 / 88; workgroups=15; avg wave life=2,211 cycles
- Empirical roofline: 0.12 TFLOPS, 15.20 GB/s
- Roofline intensity: DRAM=7.78, VL1=12.82, L2C=56.13 FLOP/Byte
- Roofline usage: HBM=0.82%, VL1=0.06%, L2C=0.05%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=4.00%, smem=3.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=18.41%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
