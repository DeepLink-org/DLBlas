# C500 Trace Profile Digest: baseline

## Kernel

- Name: `engram_gate_fwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, float*, float*, float*, float*, int, int, int, float, float, float)`
- Grid: `[16384, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `442,893,679` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 54.12% / 65.44%
- GVM share / VLS duty / L2C duty: 7.84% / 0.00% / 0.05%
- VL1/L2C hit rate: 98.22% / 39.64%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 4.00%
- WIF P25/P50/P75/max: 409.00 / 412.00 / 413.00 / 416
- DPC balance: min/avg/max=28,409,792/30,458,356/30,751,008 cycles, imbalance=7.69%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 442,893,679 | cycles | lower |
| CycleTrace instructions | 235,327 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 54.12 | % | context |
| CycleTrace BSM share | 0.4054 | % | context |
| CycleTrace GVM share | 7.838 | % | lower |
| CycleTrace ARRIVE share | 8.851 | % | lower |
| CycleTrace LDU share | 0.405 | % | context |
| CycleTrace NOP share | 13.25 | % | lower |
| Wave in flight mean | 397.7 | waves | higher |
| Wave in flight max | 416 | waves | higher |
| Wave in flight P25 | 409 | waves | context |
| Wave in flight P50 | 412 | waves | context |
| Wave in flight P75 | 413 | waves | context |
| GVM issue peak / 600 cycles | 32 | inst | lower |
| Registers/thread | 24 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 4 | % | higher |
| shared memory occupancy | 4 | % | higher |
| digest occupancy bound | 4 | % | higher |
| AP MTE duty | 65.44 | % | higher |
| AP STE duty | 13.39 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 105.7 | inst/cycle | higher |
| Instruction throughput efficiency | 25.41 | % | higher |
| Instructions per AP | 831,939 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 79.97 | % | lower |
| ISU stall cycles | 33,889,167 | cycles | lower |
| VL1 hit rate | 98.22 | % | higher |
| L2C hit rate | 39.64 | % | higher |
| DNOC read requests | 3,417,618 | req | lower |
| DNOC write requests | 1,055,317 | req | lower |
| Global memory read bytes | 436,851,296 | bytes | lower |
| Global memory write bytes | 135,072,576 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 65,484 | waves | higher |
| Dispatched waves | 66,112 | waves | context |
| Average wave life | 9,865 | cycles | lower |
| Empirical achieved FLOPS | 5.02 | TFLOPS | higher |
| Empirical achieved bandwidth | 786.1 | GB/s | higher |
| Empirical roofline intensity | 6.386 | FLOP/Byte | context |
| Roofline HBM usage | 42.65 | % | higher |
| Roofline VL1 usage | 17.69 | % | higher |
| Roofline L2C usage | 23.79 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 1.895 | FLOP/Byte | context |
| L2C-level roofline intensity | 4.579 | FLOP/Byte | context |
| DPC compute imbalance | 7.687 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 127,359 | 54.12% | Vector/Matrix Tensor Extension compute category |
| STE | 66,788 | 28.38% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 954 | 0.41% | Block shared-memory hardware category |
| GLOBAL | 18,444 | 7.84% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 20,829 | 8.85% | Synchronization/arrival category |
| LDU | 953 | 0.41% | Load data unit category |

- STE subevents by `name`: STE=27,181, S_NOP=31,180, Branch=8,427.
- GLOBAL subevents by `name`: GVM Load=15,264, GVM Store=3,180.
- ARRIVE subevents by `name`: Synchronization=20,829.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 65.44% / 13.39% / 0.00%
- Real IPC: 105.71
- VL1/L2C hit rate: 98.22% / 39.64%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=28,409,792/30,458,356/30,751,008 cycles, imbalance=7.69%
- Achieved/dispatched waves: 65,484 / 66,112; workgroups=16,528; avg wave life=9,865 cycles
- Empirical roofline: 5.02 TFLOPS, 786.13 GB/s
- Roofline intensity: DRAM=6.39, VL1=1.89, L2C=4.58 FLOP/Byte
- Roofline usage: HBM=42.65%, VL1=17.69%, L2C=23.79%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=4.00%, smem=4.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=13.25%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
