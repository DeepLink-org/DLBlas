# `head_compute_mix_bwd_kernel_opt(float const*, float const*, float const*, float const*, float*, float*, float*, int, int, int)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/head_compute_mix_bwd_run/profile-artifacts/head_compute_mix_bwd_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/head_compute_mix_bwd_run/profile-artifacts/head_compute_mix_bwd_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `head_compute_mix_bwd_kernel_opt(float const*, float const*, float const*, float const*, float*, float*, float*, int, int, int)`
- Grid / block: `[16, 1, 1]` / `[512, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/head_compute_mix_bwd_run/profile-artifacts/head_compute_mix_bwd_v0_baseline/artifacts/c-trace_output_dpc_0.json`

- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/head_compute_mix_bwd_run/profile-artifacts/head_compute_mix_bwd_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/head_compute_mix_bwd_run/profile-artifacts/head_compute_mix_bwd_v0_baseline/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag baseline \
  --cycle-dpc-id 0 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `occupancy`
- Primary: `occupancy`
- Confidence: Low
- Dominant signal: MTE share=29.07% / AP MTE duty=12.69%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=99.45%, conflict cycles/inst=0.01115; effective occupancy=2.00%
- One-line read: this kernel is classified as `occupancy` because launch occupancy is low (effective occupancy=2.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,637,173 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 29.07% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 12.69% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 99.45% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0.01115 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 2.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 13.88% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 4.00x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 13.18% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 48.43% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 2.02% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | wsm_stall / 52.07% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[16, 1, 1]` / `[512, 1, 1]`; registers/thread=15, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=2.00% and shared-memory=15.00%; digest effective bound=2.00%.
- CycleTrace WIF mean/P25/P50/P75/max=4.67/1.75/3.00/7.00/16 raw waves; P75/P25=4.00x.
- mcProfiler achieved/dispatched waves=280/284, workgroups=55, average wave life=3,663 cycles.
- DPC compute balance: min/avg/max=34,832/36,962/39,704 cycles, imbalance=13.18%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 588 | 29.07% | Vector/Matrix Tensor Extension compute category |
| STE | 1,077 | 53.24% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 33 | 1.63% | Block shared-memory hardware category |
| GLOBAL | 8 | 0.40% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 303 | 14.98% | Synchronization/arrival category |
| LDU | 14 | 0.69% | Load data unit category |

- STE subevents by `name`: STE=402, S_NOP=440, Branch=235.
- GLOBAL subevents by `name`: GVM Load=6, GVM Store=2.
- ARRIVE subevents by `name`: Synchronization=303.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 588 / 29.07% | 61,035 / 44.53% | +15.47 pp |
| STE | 1,077 / 53.24% | 31,347 / 22.87% | -30.37 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 33 / 1.63% | 4,304 / 3.14% | +1.51 pp |
| GLOBAL/GVM | 8 / 0.40% | 679 / 0.50% | +0.10 pp |
| LDU | 14 / 0.69% | 1,676 / 1.22% | +0.53 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=12.69% / 6.07% / 0.00%.
- Real IPC=8.74; instruction throughput=0.08407; throughput efficiency=2.10%; compute-instruction busy duty=14.31%.
- Instructions per AP=1,318; average cycles/instruction=2.01; average all-stage latency/instruction=38.88 cycles.
- Compute instruction cycle split: total=427,284 cycles, MTE=427,284 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=7,908 cycles, top=wsm_stall 4,118 cycles (52.07%).
- AP active cycles=516,538; average AP busy cycles=9,392; AP busy duty=0.0004566. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 4,118 cycles (52.07%)
- vls_pipeline_stall: 1,026 cycles (12.97%)
- vls_wdata_stall: 2,764 cycles (34.95%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=92.06% / 13.88% / 93.46%.
- DNOC read average latency=415.4 cycles; read/write requests=5,001/1,379; achieved bandwidth=37.24 GB/s.
- Global memory bytes read/write=351,840/167,008; global instructions read/write=384/255.
- VL1 instructions read/write=384/263; L2C instructions read/write=5,830/2,657.
- Constant read path: total=1,676, SL1=1,036, L2=69.
- DNOC latency histogram total=7,710 samples, >512-cycle samples=3,734 (48.43%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 599 samples (7.77%)
- 256-511: 3,377 samples (43.80%)
- 512-1k: 839 samples (10.88%)
- 1k-2k: 2,895 samples (37.55%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=384, write=255
- shared: read=2,368, write=1,936
- constant: read=1,676, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=384, write=263
- const_sl1c: read=1,036, write=0
- vl1c_l2c: read=5,830, write=2,657
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=2,368/1,936.
- Avg cycles per load/store=2.02/2.00; conflict cycles/inst=0.01115.
- Avg latency per load/store/atomic instruction=34.53/29.97/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 126 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.28 TFLOP/s @ 37.24 GByte/s.
- Intensities: HBM=7.42, VL1=23.59, L2C=46.99 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=2.02%, VL1=0.08%, L2C=0.13%.
- Roofline raw context: all_ops=3,849,408, all_memacs=518,848, vl1_memacs=163,164, l2c_memacs=81,920, duration=15,676, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=2.00%, smem=15.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=21.75%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=29.07% / AP MTE duty=12.69%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=99.45%, conflict cycles/inst=0.01115; effective occupancy=2.00%.
2. Likely mechanism: launch/parallelism evidence is the current primary signal: effective occupancy=2.00%, achieved/dispatched waves=280/284, and WIF P75/P25=4.00x.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=0.40%, L2C hit=13.88%, DNOC >512-cycle share=48.43%, DPC imbalance=13.18%, WIF P75/P25=4.00x, and effective occupancy=2.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: adjust grid/block coverage for this non-tensor kernel first; reduce register or shared-memory footprint only if mcTracer/compiler evidence confirms a real resource limiter.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare effective occupancy, mtreg/shared-memory occupancy, achieved waves, WIF quartiles, DPC imbalance, and `REPORT_<tag>.md`.
- Expected metric movement: effective occupancy and achieved waves should increase without creating new memory or bank-conflict pressure.

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
