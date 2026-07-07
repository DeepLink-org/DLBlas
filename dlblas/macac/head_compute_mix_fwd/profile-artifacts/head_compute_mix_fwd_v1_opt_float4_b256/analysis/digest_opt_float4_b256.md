# C500 Trace Profile Digest: opt_float4_b256

## Kernel

- Name: `head_compute_mix_fwd_kernel_opt(float const*, float const*, float const*, float const*, float*, int, int)`
- Grid: `[1024, 1, 1]`
- Block: `[256, 1, 1]`
- Span: `13,066` cycles

## Bound Classification

- Mode: `coarse`
- Type: `compute`
- Primary: `compute`

Evidence:

- MMA share / AP MMA duty: 0.00% / 0.00%
- MTE share / AP MTE duty: 68.10% / 59.28%
- GVM share / VLS duty / L2C duty: 1.18% / 0.00% / 0.01%
- VL1/L2C hit rate: 81.25% / 3.23%
- Shared-memory efficiency / conflict cycles: 0.00% / 0
- Effective occupancy bound: 2.00%
- WIF P25/P50/P75/max: 128.00 / 256.00 / 383.00 / 416
- DPC balance: min/avg/max=449,884/477,964/483,724 cycles, imbalance=7.08%

Rationale:

- MTE instruction share / duty dominates the executed compute path.
- mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.
- L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.
- ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.

## Key Metrics

| Metric | Value | Unit | Better |
|---|---:|---|---|
| Kernel span | 13,066 | cycles | lower |
| CycleTrace instructions | 1,351 | inst | context |
| CycleTrace MMA share | 0 | % | higher |
| CycleTrace MTE share | 68.1 | % | context |
| CycleTrace BSM share | 0 | % | context |
| CycleTrace GVM share | 1.184 | % | lower |
| CycleTrace ARRIVE share | 4.145 | % | lower |
| CycleTrace LDU share | 4.145 | % | context |
| CycleTrace NOP share | 6.44 | % | lower |
| Wave in flight mean | 243 | waves | higher |
| Wave in flight max | 416 | waves | higher |
| Wave in flight P25 | 128 | waves | context |
| Wave in flight P50 | 256 | waves | context |
| Wave in flight P75 | 383 | waves | context |
| GVM issue peak / 600 cycles | 5 | inst | lower |
| Registers/thread | 14 | regs | lower |
| Static shared | 0 | bytes | lower |
| mtreg occupancy | 2 | % | higher |
| shared memory occupancy | 0 | % | higher |
| digest occupancy bound | 2 | % | higher |
| AP MTE duty | 59.28 | % | higher |
| AP STE duty | 10.68 | % | higher |
| AP MMA duty | 0 | % | higher |
| Real IPC | 38.94 | inst/cycle | higher |
| Instruction throughput efficiency | 9.361 | % | higher |
| Instructions per AP | 6,228 | inst/AP | context |
| MMA compute-cycle share | 0 | % | higher |
| Top ISU stall share | 81.02 | % | lower |
| ISU stall cycles | 342,378 | cycles | lower |
| VL1 hit rate | 81.25 | % | higher |
| L2C hit rate | 3.226 | % | higher |
| DNOC read requests | 33,189 | req | lower |
| DNOC write requests | 32,804 | req | lower |
| Global memory read bytes | 4,221,152 | bytes | lower |
| Global memory write bytes | 4,195,744 | bytes | lower |
| Shared memory efficiency | 0 | % | higher |
| Avg conflict cycles/inst | 0 | cycles | lower |
| Achieved waves | 4,072 | waves | higher |
| Dispatched waves | 4,108 | waves | context |
| Average wave life | 1,755 | cycles | lower |
| Empirical achieved FLOPS | 2.308 | TFLOPS | higher |
| Empirical achieved bandwidth | 569.3 | GB/s | higher |
| Empirical roofline intensity | 4.055 | FLOP/Byte | context |
| Roofline HBM usage | 30.89 | % | higher |
| Roofline VL1 usage | 0.9388 | % | higher |
| Roofline L2C usage | 6.096 | % | higher |
| Roofline FMA peak | 14.98 | TFLOP/s | context |
| Roofline MMA-FP16 peak | 239.6 | TFLOP/s | context |
| VL1-level roofline intensity | 16.42 | FLOP/Byte | context |
| L2C-level roofline intensity | 8.217 | FLOP/Byte | context |
| DPC compute imbalance | 7.08 | % | lower |

## Instruction Distribution

Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 920 | 68.10% | Vector/Matrix Tensor Extension compute category |
| STE | 303 | 22.43% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 16 | 1.18% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 56 | 4.15% | Synchronization/arrival category |
| LDU | 56 | 4.15% | Load data unit category |

- STE subevents by `name`: STE=200, S_NOP=87, Branch=16.
- GLOBAL subevents by `name`: GVM Load=8, GVM Store=8.
- ARRIVE subevents by `name`: Synchronization=56.

## mcProfiler Hardware Evidence

- AP MTE/STE/MMA duty: 59.28% / 10.68% / 0.00%
- Real IPC: 38.94
- VL1/L2C hit rate: 81.25% / 3.23%
- Shared memory efficiency: 0.00%
- Avg conflict cycles/inst: 0
- DPC compute balance: min/avg/max=449,884/477,964/483,724 cycles, imbalance=7.08%
- Achieved/dispatched waves: 4,072 / 4,108; workgroups=1,027; avg wave life=1,755 cycles
- Empirical roofline: 2.31 TFLOPS, 569.29 GB/s
- Roofline intensity: DRAM=4.05, VL1=16.42, L2C=8.22 FLOP/Byte
- Roofline usage: HBM=30.89%, VL1=0.94%, L2C=6.10%

## Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=2.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## Scope Notes

- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.
- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.
- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
