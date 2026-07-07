# C500 Trace Profile Digest: baseline

## Kernel

- Name: `head_compute_mix_fwd_kernel_ori(float const*, float const*, float const*, float const*, float*, int, int)`
- Grid: `[2048, 1, 1]`
- Block: `[512, 1, 1]`
- Span: `27,789` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 56.52% / 53.52%
- GVM share / VLS duty / L2C duty: 2.53% / 0.00% / 0.00%
- VL1/L2C hit rate: 96.35% / 21.72%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 1.00%
- WIF P25/P50/P75/max: 303.00 / 351.00 / 365.00 / 416
- DPC balance: min/avg/max=842,240/909,010/919,948 cycles, imbalance=8.55%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 27,789 | cycles | lower |
| CycleTrace instructions | 4,857 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 56.52 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 2.532 | % | lower |
| CycleTrace ARRIVE share | 4.262 | % | lower |
| CycleTrace LDU share | 5.188 | % | context |
| CycleTrace NOP share | 10.71 | % | lower |
| Wave in flight mean | 318.9 | waves | higher |
| Wave in flight max | 416 | waves | higher |
| Wave in flight P25 | 303 | waves | context |
| Wave in flight P50 | 351 | waves | context |
| Wave in flight P75 | 365 | waves | context |
| GVM issue peak / 600 cycles | 10 | inst | lower |
| Registers/thread | 7 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 1 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 1 | % | higher |
| AP MTE duty | 53.52 | % | higher |
| AP STE duty | 16.17 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 59.6 | inst/cycle | higher |
| Instruction throughput efficiency | 14.33 | % | higher |
| Instructions per AP | 16,554 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 88.07 | % | lower |
| ISU stall cycles | 544,430 | cycles | lower |
| VL1 hit rate | 96.35 | % | higher |
| L2C hit rate | 21.72 | % | higher |
| DNOC read requests | 33,185 | req | lower |
| DNOC write requests | 32,804 | req | lower |
| Global memory read bytes | 4,220,864 | bytes | lower |
| Global memory write bytes | 4,195,712 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 16,244 | waves | higher |
| Dispatched waves | 16,396 | waves | context |
| Average wave life | 1,032 | cycles | lower |
| Empirical achieved FLOPS | 2.671 | TFLOPS | higher |
| Empirical achieved bandwidth | 327.8 | GB/s | higher |
| Empirical roofline intensity | 8.149 | FLOP/Byte | context |
| Roofline HBM usage | 17.78 | % | higher |
| Roofline VL1 usage | 3.242 | % | higher |
| Roofline L2C usage | 5.268 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 5.502 | FLOP/Byte | context |
| L2C-level roofline intensity | 11 | FLOP/Byte | context |
| DPC compute imbalance | 8.549 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 2,745 | 56.52% | Vector/Matrix Tensor Extension compute category |
| STE | 1,530 | 31.50% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 123 | 2.53% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 207 | 4.26% | Synchronization/arrival category |
| LDU | 252 | 5.19% | Load data unit category |

- STE subevents by `name`: STE=965, S_NOP=520, Branch=42.
- GLOBAL subevents by `name`: GVM Load=84, GVM Store=39.
- ARRIVE subevents by `name`: Synchronization=207.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 53.52% / 16.17% / 0.00%
- Real IPC: 59.60
- VL1/L2C hit rate: 96.35% / 21.72%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=842,240/909,010/919,948 cycles, imbalance=8.55%
- Achieved/dispatched waves: 16,244 / 16,396; workgroups=2,051; avg wave life=1,032 cycles
- Empirical roofline: 2.67 TFLOPS, 327.79 GB/s
- Roofline intensity: DRAM=8.15, VL1=5.50, L2C=11.00 FLOP/Byte
- Roofline usage: HBM=17.78%, VL1=3.24%, L2C=5.27%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=1.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=10.71%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
