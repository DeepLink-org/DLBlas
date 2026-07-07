# `norm_fn_kernel_opt(float const*, float const*, float*, int, int, int, float)` Trace Profiling Report

**Tag:** `final`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v9_final`
**Artifact directory:** `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v9_final/artifacts`

## 0. Profiling setup

- Kernel: `norm_fn_kernel_opt(float const*, float const*, float*, int, int, int, float)`
- Grid / block: `[24, 13, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v9_final/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v9_final/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v9_final/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag final \
  --cycle-dpc-id 2,3 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `compute`
- Primary: `compute`
- Confidence: Medium
- Dominant signal: MTE share=39.80% / AP MTE duty=14.56%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=3.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=39.80%, AP MTE duty=14.56%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 2,048,215 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 39.80% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 14.56% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 3.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 83.87% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 6.18x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 11.46% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 54.60% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 5.32% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 88.32% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[24, 13, 1]` / `[256, 1, 1]`; registers/thread=22, static shared=2,048 bytes.
- mcTracer occupancy fields are mtreg=4.00% and shared-memory=3.00%; digest effective bound=3.00%.
- CycleTrace WIF mean/P25/P50/P75/max=62.47/17.00/54.00/105.00/156 raw waves; P75/P25=6.18x.
- mcProfiler achieved/dispatched waves=1,600/1,616, workgroups=404, average wave life=3,682 cycles.
- DPC compute balance: min/avg/max=138,144/151,246/155,484 cycles, imbalance=11.46%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 468 | 39.80% | Vector/Matrix Tensor Extension compute category |
| STE | 537 | 45.66% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 3 | 0.26% | Block shared-memory hardware category |
| GLOBAL | 30 | 2.55% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 123 | 10.46% | Synchronization/arrival category |
| LDU | 15 | 1.28% | Load data unit category |

- STE subevents by `name`: STE=255, S_NOP=204, Branch=78.
- GLOBAL subevents by `name`: GVM Load=30, GVM Store=0.
- ARRIVE subevents by `name`: Synchronization=123.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 468 / 39.80% | 281,782 / 53.12% | +13.32 pp |
| STE | 537 / 45.66% | 122,322 / 23.06% | -22.60 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 3 / 0.26% | 9,888 / 1.86% | +1.61 pp |
| GLOBAL/GVM | 30 / 2.55% | 13,407 / 2.53% | -0.02 pp |
| LDU | 15 / 1.28% | 8,016 / 1.51% | +0.24 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=14.56% / 6.23% / 0.00%.
- Real IPC=24.09; instruction throughput=0.2316; throughput efficiency=5.79%; compute-instruction busy duty=15.40%.
- Instructions per AP=5,101; average cycles/instruction=4.00; average all-stage latency/instruction=45.74 cycles.
- Compute instruction cycle split: total=1,470,352 cycles, MTE=1,470,352 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=221,008 cycles, top=vls_pipeline_stall 195,196 cycles (88.32%).
- AP active cycles=1,963,724; average AP busy cycles=19,065; AP busy duty=0.0006417. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 12,704 cycles (5.75%)
- vls_pipeline_stall: 195,196 cycles (88.32%)
- vls_wdata_stall: 13,108 cycles (5.93%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=86.77% / 83.87% / 97.92%.
- DNOC read average latency=1,145 cycles; read/write requests=12,171/6,002; achieved bandwidth=98.09 GB/s.
- Global memory bytes read/write=1,157,248/763,008; global instructions read/write=12,360/1,047.
- VL1 instructions read/write=12,360/1,051; L2C instructions read/write=108,782/12,235.
- Constant read path: total=8,016, SL1=5,008, L2=104.
- DNOC latency histogram total=21,988 samples, >512-cycle samples=12,005 (54.60%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 5,530 samples (25.15%)
- 256-511: 4,453 samples (20.25%)
- 512-1k: 5,603 samples (25.48%)
- 1k-2k: 3,915 samples (17.81%)
- 2k-: 2,487 samples (11.31%)

Memory data flow:

- global: read=12,360, write=1,047
- shared: read=5,871, write=4,017
- constant: read=8,016, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=12,360, write=1,051
- const_sl1c: read=5,008, write=0
- vl1c_l2c: read=108,782, write=12,235
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=5,871/4,017.
- Avg cycles per load/store=4.00/4.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=41.23/33.88/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 1.04 TFLOP/s @ 98.09 GByte/s.
- Intensities: HBM=10.55, VL1=6.04, L2C=1.60 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=5.32%, VL1=1.14%, L2C=14.03%.
- Roofline raw context: all_ops=20,264,320, all_memacs=1,920,256, vl1_memacs=3,354,436, l2c_memacs=12,656,640, duration=22,024, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=4.00%, smem=3.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=17.35%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=39.80% / AP MTE duty=14.56%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=3.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=39.80%, MMA share=0.00%, AP MTE duty=14.56%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=2.55%, L2C hit=83.87%, DNOC >512-cycle share=54.60%, DPC imbalance=11.46%, WIF P75/P25=6.18x, and effective occupancy=3.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: focus the compute path indicated by SOL and compute-cycle split, reducing scalar/vector overhead or improving the intended compute instruction mix.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare AP duty ratios, CycleTrace instruction shares, compute instruction cycle split, shared memory efficiency, and `REPORT_<tag>.md`.
- Expected metric movement: the intended compute duty and compute-cycle share should increase without regressing memory hierarchy or occupancy metrics.

## 7. Artifacts

- Full metrics: `analysis/metrics_all_final.json`
- Key metrics: `analysis/metrics_key_final.json`
- Digest: `analysis/digest_final.md`
- This report: `REPORT_final.md`

## 8. Caveats

- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.
- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.
