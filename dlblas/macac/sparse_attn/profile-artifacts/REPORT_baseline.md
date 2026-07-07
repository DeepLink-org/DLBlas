# `sparse_attn_kernel_opt(unsigned short const*, unsigned short const*, int const*, float const*, unsigned short*, int, int, int, int, int, int, float)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `sparse_attn_kernel_opt(unsigned short const*, unsigned short const*, int const*, float const*, unsigned short*, int, int, int, int, int, int, float)`
- Grid / block: `[256, 1, 1]` / `[64, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=39.74% / AP MTE duty=10.68%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=12.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=39.74%, AP MTE duty=10.68%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,465,915 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 39.74% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 10.68% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 12.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 82.75% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 10.00x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 9.17% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 25.93% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 0.36% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | wsm_stall / 76.71% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[256, 1, 1]` / `[64, 1, 1]`; registers/thread=66, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=12.00% and shared-memory=0.00%; digest effective bound=12.00%.
- CycleTrace WIF mean/P25/P50/P75/max=11.33/2.00/8.00/20.00/32 raw waves; P75/P25=10.00x.
- mcProfiler achieved/dispatched waves=366/368, workgroups=284, average wave life=9,641 cycles.
- DPC compute balance: min/avg/max=300,032/324,390/329,792 cycles, imbalance=9.17%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 2,180 | 39.74% | Vector/Matrix Tensor Extension compute category |
| STE | 2,332 | 42.51% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 320 | 5.83% | Block shared-memory hardware category |
| GLOBAL | 49 | 0.89% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 549 | 10.01% | Synchronization/arrival category |
| LDU | 56 | 1.02% | Load data unit category |

- STE subevents by `name`: STE=980, S_NOP=894, Branch=458.
- GLOBAL subevents by `name`: GVM Load=48, GVM Store=1.
- ARRIVE subevents by `name`: Synchronization=549.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 2,180 / 39.74% | 569,521 / 48.01% | +8.27 pp |
| STE | 2,332 / 42.51% | 252,050 / 21.25% | -21.26 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 320 / 5.83% | 80,960 / 6.82% | +0.99 pp |
| GLOBAL/GVM | 49 / 0.89% | 12,549 / 1.06% | +0.16 pp |
| LDU | 56 / 1.02% | 14,784 / 1.25% | +0.23 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=10.68% / 4.60% / 0.00%.
- Real IPC=17.59; instruction throughput=0.1691; throughput efficiency=4.23%; compute-instruction busy duty=11.83%.
- Instructions per AP=11,407; average cycles/instruction=2.00; average all-stage latency/instruction=38.30 cycles.
- Compute instruction cycle split: total=3,614,772 cycles, MTE=3,614,772 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=92,567 cycles, top=wsm_stall 71,008 cycles (76.71%).
- AP active cycles=5,484,142; average AP busy cycles=53,244; AP busy duty=0.001965. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 71,008 cycles (76.71%)
- vls_pipeline_stall: 18,831 cycles (20.34%)
- vls_wdata_stall: 2,728 cycles (2.95%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=98.21% / 82.75% / 96.97%.
- DNOC read average latency=433.2 cycles; read/write requests=3,683/1,171; achieved bandwidth=6.66 GB/s.
- Global memory bytes read/write=255,968/143,168; global instructions read/write=12,144/357.
- VL1 instructions read/write=12,192/360; L2C instructions read/write=26,403/2,266.
- Constant read path: total=14,784, SL1=8,996, L2=301.
- DNOC latency histogram total=3,891 samples, >512-cycle samples=1,009 (25.93%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 511 samples (13.13%)
- 256-511: 2,371 samples (60.94%)
- 512-1k: 1,009 samples (25.93%)
- 1k-2k: 0 samples (0.00%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=12,144, write=357
- shared: read=52,832, write=28,336
- constant: read=14,784, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=12,192, write=360
- const_sl1c: read=8,996, write=0
- vl1c_l2c: read=26,403, write=2,266
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=52,832/28,336.
- Avg cycles per load/store=1.99/2.01; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.50/29.77/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.61 TFLOP/s @ 6.66 GByte/s.
- Intensities: HBM=91.09, VL1=11.32, L2C=23.30 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=0.36%, VL1=0.36%, L2C=0.56%.
- Roofline raw context: all_ops=36,357,568, all_memacs=399,136, vl1_memacs=3,212,296, l2c_memacs=1,560,576, duration=67,452, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=12.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=16.30%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=39.74% / AP MTE duty=10.68%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=12.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=39.74%, MMA share=0.00%, AP MTE duty=10.68%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=0.89%, L2C hit=82.75%, DNOC >512-cycle share=25.93%, DPC imbalance=9.17%, WIF P75/P25=10.00x, and effective occupancy=12.00%.
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
