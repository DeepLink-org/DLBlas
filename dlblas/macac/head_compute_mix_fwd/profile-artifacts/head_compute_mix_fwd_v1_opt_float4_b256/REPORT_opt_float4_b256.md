# `head_compute_mix_fwd_kernel_opt(float const*, float const*, float const*, float const*, float*, int, int)` Trace Profiling Report

**Tag:** `opt_float4_b256`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v1_opt_float4_b256`
**Artifact directory:** `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v1_opt_float4_b256/artifacts`

## 0. Profiling setup

- Kernel: `head_compute_mix_fwd_kernel_opt(float const*, float const*, float const*, float const*, float*, int, int)`
- Grid / block: `[1024, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v1_opt_float4_b256/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v1_opt_float4_b256/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v1_opt_float4_b256/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag opt_float4_b256 \
  --cycle-dpc-id 2,3 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `compute`
- Primary: `compute`
- Confidence: Medium
- Dominant signal: MTE share=68.10% / AP MTE duty=59.28%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=2.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=68.10%, AP MTE duty=59.28%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 13,066 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 68.10% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 59.28% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 2.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 3.23% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 2.99x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.08% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 0.00% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 30.89% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 81.02% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[1024, 1, 1]` / `[256, 1, 1]`; registers/thread=14, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=2.00% and shared-memory=0.00%; digest effective bound=2.00%.
- CycleTrace WIF mean/P25/P50/P75/max=242.97/128.00/256.00/383.00/416 raw waves; P75/P25=2.99x.
- mcProfiler achieved/dispatched waves=4,072/4,108, workgroups=1,027, average wave life=1,755 cycles.
- DPC compute balance: min/avg/max=449,884/477,964/483,724 cycles, imbalance=7.08%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

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

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 920 / 68.10% | 468,369 / 72.31% | +4.21 pp |
| STE | 303 / 22.43% | 102,042 / 15.75% | -6.67 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 16 / 1.18% | 8,123 / 1.25% | +0.07 pp |
| LDU | 56 / 4.15% | 28,452 / 4.39% | +0.25 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=59.28% / 10.68% / 0.00%.
- Real IPC=38.94; instruction throughput=0.3745; throughput efficiency=9.36%; compute-instruction busy duty=100.06%.
- Instructions per AP=6,228; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=10,066,196 cycles, MTE=10,066,196 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=342,378 cycles, top=vls_pipeline_stall 277,394 cycles (81.02%).
- AP active cycles=955,379; average AP busy cycles=9,276; AP busy duty=0.0004832. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 277,394 cycles (81.02%)
- vls_wdata_stall: 64,984 cycles (18.98%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=81.25% / 3.23% / 99.23%.
- DNOC read average latency=316.6 cycles; read/write requests=33,189/32,804; achieved bandwidth=569.29 GB/s.
- Global memory bytes read/write=4,221,152/4,195,744; global instructions read/write=4,056/4,067.
- VL1 instructions read/write=4,060/4,059; L2C instructions read/write=35,389/65,572.
- Constant read path: total=28,452, SL1=17,484, L2=134.
- DNOC latency histogram total=44,731 samples, >512-cycle samples=0 (0.00%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 32,699 samples (73.10%)
- 256-511: 12,032 samples (26.90%)
- 512-1k: 0 samples (0.00%)
- 1k-2k: 0 samples (0.00%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=4,056, write=4,067
- shared: read=0, write=0
- constant: read=28,452, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=4,060, write=4,059
- const_sl1c: read=17,484, write=0
- vl1c_l2c: read=35,389, write=65,572
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 2.31 TFLOP/s @ 569.29 GByte/s.
- Intensities: HBM=4.05, VL1=16.42, L2C=8.22 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=30.89%, VL1=0.94%, L2C=6.10%.
- Roofline raw context: all_ops=34,128,448, all_memacs=8,416,896, vl1_memacs=2,078,732, l2c_memacs=4,153,344, duration=16,633, core_clk=1.125 GHz, mc_clk=0.900 GHz.
- Roofline peak and usage fields are reported as mcProfiler chart context; operator-theoretical FLOP/Byte still requires external shape/formula metadata.

### 3.7 Data Availability Boundaries
- Current artifacts do not provide per-source-line or per-PC stall attribution; the report cannot name source-line hotspots.
- Current artifacts do not provide PM-sampling or per-AP utilization time series; DPC cycles and WIF quartiles are aggregate balance proxies only.
- Current artifacts do not expose a C500 sectors/request or useful-bytes/sector equivalent; memory flow and hit rates are not enough to claim global coalescing quality.

### 3.8 Metric Coverage

| Category | Count | Meaning |
|---|---:|---|
| Reported metric groups | 9 | Metric groups directly shown in `REPORT_<tag>.md` |
| Parsed but not promoted groups | 8 | Available in `metrics_all_<tag>.json`, omitted from the main prose unless diagnostic |
| Unavailable dimensions | 5 | Analyses that current artifacts cannot support |

## 4. Diagnosis

### MEDIUM: Low launch occupancy bound

- Evidence: mtreg=2.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## 5. Inference Chain

1. Measured: MTE share=68.10% / AP MTE duty=59.28%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=2.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=68.10%, MMA share=0.00%, AP MTE duty=59.28%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=1.18%, L2C hit=3.23%, DNOC >512-cycle share=0.00%, DPC imbalance=7.08%, WIF P75/P25=2.99x, and effective occupancy=2.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: focus the compute path indicated by SOL and compute-cycle split, reducing scalar/vector overhead or improving the intended compute instruction mix.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare AP duty ratios, CycleTrace instruction shares, compute instruction cycle split, shared memory efficiency, and `REPORT_<tag>.md`.
- Expected metric movement: the intended compute duty and compute-cycle share should increase without regressing memory hierarchy or occupancy metrics.

## 7. Artifacts

- Full metrics: `analysis/metrics_all_opt_float4_b256.json`
- Key metrics: `analysis/metrics_key_opt_float4_b256.json`
- Digest: `analysis/digest_opt_float4_b256.md`
- This report: `REPORT_opt_float4_b256.md`

## 8. Caveats

- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.
- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.
