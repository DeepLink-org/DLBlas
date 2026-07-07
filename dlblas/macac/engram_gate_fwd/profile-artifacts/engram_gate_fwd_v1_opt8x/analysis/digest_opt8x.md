# C500 Trace Profile Digest: opt8x

## Kernel

- Name: `engram_gate_fwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, float*, float*, float*, float*, int, int, int, float, float, float)`
- Grid: `[16384, 1, 1]`
- Block: `[64, 1, 1]`
- Span: `420,988,572` cycles

## Bound Classification

- Mode: `coarse`
- Type: `memory`
- Primary: `memory`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 66.18% / 73.66%
- GVM share / VLS duty / L2C duty: 12.94% / 0.00% / 0.05%
- VL1/L2C hit rate: 98.21% / 38.98%
- Shared-memory efficiency / conflict cycles: 100.00% / 0
- Effective occupancy bound: 1.00%
- WIF P25/P50/P75/max: 403.00 / 407.00 / 410.00 / 416
- DPC balance: min/avg/max=18,457,872/19,788,446/19,978,528 cycles, imbalance=7.68%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- Memory pressure threshold fired from GVM/VLS/L2C/cache evidence.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 420,988,572 | cycles | lower |
| CycleTrace instructions | 136,242 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 66.18 | % | context |
| CycleTrace BSM share | 1.718 | % | context |
| CycleTrace GVM share | 12.94 | % | lower |
| CycleTrace ARRIVE share | 8.072 | % | lower |
| CycleTrace LDU share | 0.171 | % | context |
| CycleTrace NOP share | 6.248 | % | lower |
| Wave in flight mean | 355.7 | waves | higher |
| Wave in flight max | 416 | waves | higher |
| Wave in flight P25 | 403 | waves | context |
| Wave in flight P50 | 407 | waves | context |
| Wave in flight P75 | 410 | waves | context |
| GVM issue peak / 600 cycles | 60 | inst | lower |
| Registers/thread | 60 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 11 | % | higher |
| shared memory occupancy | 1 | % | higher |
| digest occupancy bound | 1 | % | higher |
| AP MTE duty | 73.66 | % | higher |
| AP STE duty | 3.91 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 103.9 | inst/cycle | higher |
| Instruction throughput efficiency | 24.98 | % | higher |
| Instructions per AP | 512,542 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 84.46 | % | lower |
| ISU stall cycles | 34,120,304 | cycles | lower |
| VL1 hit rate | 98.21 | % | higher |
| L2C hit rate | 38.98 | % | higher |
| DNOC read requests | 3,445,230 | req | lower |
| DNOC write requests | 1,055,335 | req | lower |
| Global memory read bytes | 440,762,112 | bytes | lower |
| Global memory write bytes | 135,073,280 | bytes | lower |
| Shared memory efficiency | 100 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 16,800 | waves | higher |
| Dispatched waves | 16,960 | waves | context |
| Average wave life | 22,137 | cycles | lower |
| Empirical achieved FLOPS | 5.801 | TFLOPS | higher |
| Empirical achieved bandwidth | 1,263 | GB/s | higher |
| Empirical roofline intensity | 4.593 | FLOP/Byte | context |
| Roofline HBM usage | 68.53 | % | higher |
| Roofline VL1 usage | 27.51 | % | higher |
| Roofline L2C usage | 37.97 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 1.408 | FLOP/Byte | context |
| L2C-level roofline intensity | 3.316 | FLOP/Byte | context |
| DPC compute imbalance | 7.685 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 90,168 | 66.18% | Vector/Matrix Tensor Extension compute category |
| STE | 14,875 | 10.92% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 2,340 | 1.72% | Block shared-memory hardware category |
| GLOBAL | 17,628 | 12.94% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 10,998 | 8.07% | Synchronization/arrival category |
| LDU | 233 | 0.17% | Load data unit category |

- STE subevents by `name`: STE=4,764, S_NOP=8,512, Branch=1,599.
- GLOBAL subevents by `name`: GVM Load=14,976, GVM Store=2,652.
- ARRIVE subevents by `name`: Synchronization=10,998.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 73.66% / 3.91% / 0.00%
- Real IPC: 103.93
- VL1/L2C hit rate: 98.21% / 38.98%
- Shared memory efficiency: 100.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=18,457,872/19,788,446/19,978,528 cycles, imbalance=7.68%
- Achieved/dispatched waves: 16,800 / 16,960; workgroups=16,528; avg wave life=22,137 cycles
- Empirical roofline: 5.80 TFLOPS, 1263.10 GB/s
- Roofline intensity: DRAM=4.59, VL1=1.41, L2C=3.32 FLOP/Byte
- Roofline usage: HBM=68.53%, VL1=27.51%, L2C=37.97%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=11.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
