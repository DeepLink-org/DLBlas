# C500 Trace Profile Digest: baseline

## Kernel

- Name: `engram_gate_bwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, float*, float*, float*, int, int, int, float, float)`
- Grid: `[56, 1, 1]`
- Block: `[128, 1, 1]`
- Span: `1,869,755` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 50.18% / 8.19%
- GVM share / VLS duty / L2C duty: 3.83% / 0.00% / 0.00%
- VL1/L2C hit rate: 96.51% / 50.05%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 2.00%
- WIF P25/P50/P75/max: 1.00 / 3.00 / 5.25 / 14
- DPC balance: min/avg/max=38,188/40,779/42,156 cycles, imbalance=9.73%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- NOP/IPC thresholds indicate latency or pipeline-bubble pressure.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 1,869,755 | cycles | lower |
| CycleTrace instructions | 548 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 50.18 | % | context |
| CycleTrace BSM share | 1.46 | % | context |
| CycleTrace GVM share | 3.832 | % | lower |
| CycleTrace ARRIVE share | 12.96 | % | lower |
| CycleTrace LDU share | 1.46 | % | context |
| CycleTrace NOP share | 12.23 | % | lower |
| Wave in flight mean | 4.059 | waves | higher |
| Wave in flight max | 14 | waves | higher |
| Wave in flight P25 | 1 | waves | context |
| Wave in flight P50 | 3 | waves | context |
| Wave in flight P75 | 5.25 | waves | context |
| GVM issue peak / 600 cycles | 8 | inst | lower |
| Registers/thread | 26 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 5 | % | higher |
| shared memory occupancy | 2 | % | higher |
| digest occupancy bound | 2 | % | higher |
| AP MTE duty | 8.187 | % | higher |
| AP STE duty | 1.835 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 10.13 | inst/cycle | higher |
| Instruction throughput efficiency | 2.435 | % | higher |
| Instructions per AP | 922.2 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 37.69 | % | lower |
| ISU stall cycles | 12,820 | cycles | lower |
| VL1 hit rate | 96.51 | % | higher |
| L2C hit rate | 50.05 | % | higher |
| DNOC read requests | 5,113 | req | lower |
| DNOC write requests | 1,443 | req | lower |
| Global memory read bytes | 350,208 | bytes | lower |
| Global memory write bytes | 172,800 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 266 | waves | higher |
| Dispatched waves | 268 | waves | context |
| Average wave life | 2,859 | cycles | lower |
| Empirical achieved FLOPS | 0.4454 | TFLOPS | higher |
| Empirical achieved bandwidth | 62.14 | GB/s | higher |
| Empirical roofline intensity | 7.167 | FLOP/Byte | context |
| Roofline HBM usage | 3.372 | % | higher |
| Roofline VL1 usage | 0.5024 | % | higher |
| Roofline L2C usage | 0.5915 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 5.92 | FLOP/Byte | context |
| L2C-level roofline intensity | 16.34 | FLOP/Byte | context |
| DPC compute imbalance | 9.73 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 275 | 50.18% | Vector/Matrix Tensor Extension compute category |
| STE | 165 | 30.11% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 8 | 1.46% | Block shared-memory hardware category |
| GLOBAL | 21 | 3.83% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 71 | 12.96% | Synchronization/arrival category |
| LDU | 8 | 1.46% | Load data unit category |

- STE subevents by `name`: STE=78, S_NOP=67, Branch=20.
- GLOBAL subevents by `name`: GVM Load=16, GVM Store=2.
- ARRIVE subevents by `name`: Synchronization=71.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 8.19% / 1.83% / 0.00%
- Real IPC: 10.13
- VL1/L2C hit rate: 96.51% / 50.05%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=38,188/40,779/42,156 cycles, imbalance=9.73%
- Achieved/dispatched waves: 266 / 268; workgroups=95; avg wave life=2,859 cycles
- Empirical roofline: 0.45 TFLOPS, 62.14 GB/s
- Roofline intensity: DRAM=7.17, VL1=5.92, L2C=16.34 FLOP/Byte
- Roofline usage: HBM=3.37%, VL1=0.50%, L2C=0.59%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=5.00%, smem=2.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=12.23%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
