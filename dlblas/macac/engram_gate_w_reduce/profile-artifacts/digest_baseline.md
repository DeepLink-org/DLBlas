# C500 Trace Profile Digest: baseline

## Kernel

- Name: `_Z31engram_gate_w_reduce_kernel_oriPKfPKDF16_S2_S0_S0_PfS3_iii`
- Grid: `[64, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `115,187,543` cycles

## Bound Classification

- Mode: `coarse`
- Type: `memory`
- Primary: `memory`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 52.12% / 11.01%
- GVM share / VLS duty / L2C duty: 16.10% / 0.00% / 2.23%
- VL1/L2C hit rate: 96.85% / 2.48%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 5.00%
- WIF P25/P50/P75/max: 8.00 / 16.00 / 24.00 / 32
- DPC balance: min/avg/max=87,676,032/94,066,612/94,979,552 cycles, imbalance=7.76%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- Memory pressure threshold fired from GVM/VLS/L2C/cache evidence.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 115,187,543 | cycles | lower |
| CycleTrace instructions | 875,088 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 52.12 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 16.1 | % | lower |
| CycleTrace ARRIVE share | 12.71 | % | lower |
| CycleTrace LDU share | 0.8475 | % | context |
| CycleTrace NOP share | 5.65 | % | lower |
| Wave in flight mean | 15.99 | waves | higher |
| Wave in flight max | 32 | waves | higher |
| Wave in flight P25 | 8 | waves | context |
| Wave in flight P50 | 16 | waves | context |
| Wave in flight P75 | 24 | waves | context |
| GVM issue peak / 600 cycles | 16 | inst | lower |
| Registers/thread | 26 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 5 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 5 | % | higher |
| AP MTE duty | 11.01 | % | higher |
| AP STE duty | 2.028 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 19.86 | inst/cycle | higher |
| Instruction throughput efficiency | 4.774 | % | higher |
| Instructions per AP | 3,279,129 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 95.52 | % | lower |
| ISU stall cycles | 91,228,964 | cycles | lower |
| VL1 hit rate | 96.85 | % | higher |
| L2C hit rate | 2.48 | % | higher |
| DNOC read requests | 114,007,502 | req | lower |
| DNOC write requests | 2,083,735 | req | lower |
| Global memory read bytes | 14,586,605,856 | bytes | lower |
| Global memory write bytes | 263,828,576 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 510,056 | waves | higher |
| Dispatched waves | 515,008 | waves | context |
| Average wave life | 3,294 | cycles | lower |
| Empirical achieved FLOPS | 0.793 | TFLOPS | higher |
| Empirical achieved bandwidth | 972.8 | GB/s | higher |
| Empirical roofline intensity | 0.8151 | FLOP/Byte | context |
| Roofline HBM usage | 52.78 | % | higher |
| Roofline VL1 usage | 6.506 | % | higher |
| Roofline L2C usage | 20.59 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 0.8139 | FLOP/Byte | context |
| L2C-level roofline intensity | 0.8359 | FLOP/Byte | context |
| DPC compute imbalance | 7.764 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 456,084 | 52.12% | Vector/Matrix Tensor Extension compute category |
| STE | 159,444 | 18.22% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 140,904 | 16.10% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 111,240 | 12.71% | Synchronization/arrival category |
| LDU | 7,416 | 0.85% | Load data unit category |

- STE subevents by `name`: STE=84,048, S_NOP=49,440, Branch=25,956.
- GLOBAL subevents by `name`: GVM Load=138,432, GVM Store=2,472.
- ARRIVE subevents by `name`: Synchronization=111,240.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 11.01% / 2.03% / 0.00%
- Real IPC: 19.86
- VL1/L2C hit rate: 96.85% / 2.48%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=87,676,032/94,066,612/94,979,552 cycles, imbalance=7.76%
- Achieved/dispatched waves: 510,056 / 515,008; workgroups=128,752; avg wave life=3,294 cycles
- Empirical roofline: 0.79 TFLOPS, 972.84 GB/s
- Roofline intensity: DRAM=0.82, VL1=0.81, L2C=0.84 FLOP/Byte
- Roofline usage: HBM=52.78%, VL1=6.51%, L2C=20.59%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=5.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
