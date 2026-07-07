# `head_compute_mix_fwd_kernel_ori(float const*, float const*, float const*, float const*, float*, int, int)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v0_baseline`
**Artifact directory:** `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `head_compute_mix_fwd_kernel_ori(float const*, float const*, float const*, float const*, float*, int, int)`
- Grid / block: `[2048, 1, 1]` / `[512, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/head_compute_mix_fwd_run/profile-artifacts/head_compute_mix_fwd_v0_baseline/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag baseline \
  --cycle-dpc-id 2,3 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `compute`
- Primary: `compute`
- Confidence: Medium
- Dominant signal: MTE share=56.52% / AP MTE duty=53.52%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=1.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=56.52%, AP MTE duty=53.52%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 27,789 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 56.52% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 53.52% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 1.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 21.72% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 1.20x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 8.55% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 0.02% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 17.78% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 88.07% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[2048, 1, 1]` / `[512, 1, 1]`; registers/thread=7, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=1.00% and shared-memory=0.00%; digest effective bound=1.00%.
- CycleTrace WIF mean/P25/P50/P75/max=318.87/303.00/351.00/365.00/416 raw waves; P75/P25=1.20x.
- mcProfiler achieved/dispatched waves=16,244/16,396, workgroups=2,051, average wave life=1,032 cycles.
- DPC compute balance: min/avg/max=842,240/909,010/919,948 cycles, imbalance=8.55%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

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

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 2,745 / 56.52% | 1,087,941 / 63.19% | +6.68 pp |
| STE | 1,530 / 31.50% | 373,410 / 21.69% | -9.81 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 123 / 2.53% | 48,699 / 2.83% | +0.30 pp |
| LDU | 252 / 5.19% | 97,356 / 5.65% | +0.47 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=53.52% / 16.17% / 0.00%.
- Real IPC=59.60; instruction throughput=0.5731; throughput efficiency=14.33%; compute-instruction busy duty=78.71%.
- Instructions per AP=16,554; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=16,637,588 cycles, MTE=16,637,588 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=544,430 cycles, top=vls_pipeline_stall 479,478 cycles (88.07%).
- AP active cycles=2,309,709; average AP busy cycles=22,424; AP busy duty=0.0008405. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 479,478 cycles (88.07%)
- vls_wdata_stall: 64,952 cycles (11.93%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.35% / 21.72% / 99.78%.
- DNOC read average latency=87.12 cycles; read/write requests=33,185/32,804; achieved bandwidth=327.79 GB/s.
- Global memory bytes read/write=4,220,864/4,195,712; global instructions read/write=32,496/16,251.
- VL1 instructions read/write=32,464/16,235; L2C instructions read/write=51,499/65,572.
- Constant read path: total=97,356, SL1=59,876, L2=130.
- DNOC latency histogram total=33,184 samples, >512-cycle samples=6 (0.02%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 32,807 samples (98.86%)
- 256-511: 371 samples (1.12%)
- 512-1k: 6 samples (0.02%)
- 1k-2k: 0 samples (0.00%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=32,496, write=16,251
- shared: read=0, write=0
- constant: read=97,356, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=32,464, write=16,235
- const_sl1c: read=59,876, write=0
- vl1c_l2c: read=51,499, write=65,572
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 2.67 TFLOP/s @ 327.79 GByte/s.
- Intensities: HBM=8.15, VL1=5.50, L2C=11.00 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=17.78%, VL1=3.24%, L2C=5.27%.
- Roofline raw context: all_ops=68,584,768, all_memacs=8,416,576, vl1_memacs=12,466,188, l2c_memacs=6,233,088, duration=28,886, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=1.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=10.71%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=56.52% / AP MTE duty=53.52%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=1.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=56.52%, MMA share=0.00%, AP MTE duty=53.52%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=2.53%, L2C hit=21.72%, DNOC >512-cycle share=0.02%, DPC imbalance=8.55%, WIF P75/P25=1.20x, and effective occupancy=1.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: focus the compute path indicated by SOL and compute-cycle split, reducing scalar/vector overhead or improving the intended compute instruction mix.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare AP duty ratios, CycleTrace instruction shares, compute instruction cycle split, shared memory efficiency, and `REPORT_<tag>.md`.
- Expected metric movement: the intended compute duty and compute-cycle share should increase without regressing memory hierarchy or occupancy metrics.

## 7. Artifacts

- Full metrics: `analysis/metrics_all_baseline.json`
- Key metrics: `analysis/metrics_key_baseline.json`
- Digest: `analysis/digest_baseline.md`
- This report: `REPORT_baseline.md`

## 8. Caveats

- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.
- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.
