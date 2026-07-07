# `apply_mix_kernel_opt(unsigned short const*, float const*, unsigned short*, int, int, int, int)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/apply_mix_run/profile-artifacts/apply_mix_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/apply_mix_run/profile-artifacts/apply_mix_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `apply_mix_kernel_opt(unsigned short const*, float const*, unsigned short*, int, int, int, int)`
- Grid / block: `[2048, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/apply_mix_run/profile-artifacts/apply_mix_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/apply_mix_run/profile-artifacts/apply_mix_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/apply_mix_run/profile-artifacts/apply_mix_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=51.97% / AP MTE duty=41.75%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=5.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=51.97%, AP MTE duty=41.75%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 29,689,532 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 51.97% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 41.75% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 5.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 44.82% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 1.36x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.06% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 22.01% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 39.49% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 93.18% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[2048, 1, 1]` / `[256, 1, 1]`; registers/thread=26, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=5.00% and shared-memory=0.00%; digest effective bound=5.00%.
- CycleTrace WIF mean/P25/P50/P75/max=282.31/253.00/320.00/345.00/416 raw waves; P75/P25=1.36x.
- mcProfiler achieved/dispatched waves=8,144/8,224, workgroups=2,056, average wave life=2,033 cycles.
- DPC compute balance: min/avg/max=543,824/579,608/584,720 cycles, imbalance=7.06%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 3,135 | 51.97% | Vector/Matrix Tensor Extension compute category |
| STE | 1,682 | 27.88% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 513 | 8.50% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 589 | 9.76% | Synchronization/arrival category |
| LDU | 113 | 1.87% | Load data unit category |

- STE subevents by `name`: STE=1,175, S_NOP=412, Branch=95.
- GLOBAL subevents by `name`: GVM Load=456, GVM Store=57.
- ARRIVE subevents by `name`: Synchronization=589.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 3,135 / 51.97% | 1,157,120 / 54.16% | +2.18 pp |
| STE | 1,682 / 27.88% | 487,664 / 22.82% | -5.06 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 513 / 8.50% | 182,732 / 8.55% | +0.05 pp |
| LDU | 113 / 1.87% | 48,832 / 2.29% | +0.41 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=41.75% / 17.57% / 0.00%.
- Real IPC=52.48; instruction throughput=0.5046; throughput efficiency=12.62%; compute-instruction busy duty=41.78%.
- Instructions per AP=20,544; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=4,649,152 cycles, MTE=4,649,152 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=1,196,213 cycles, top=vls_pipeline_stall 1,114,581 cycles (93.18%).
- AP active cycles=2,774,858; average AP busy cycles=26,940; AP busy duty=0.001175. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 1,114,581 cycles (93.18%)
- vls_wdata_stall: 81,632 cycles (6.82%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.53% / 44.82% / 95.69%.
- DNOC read average latency=374 cycles; read/write requests=165,301/41,247; achieved bandwidth=727.91 GB/s.
- Global memory bytes read/write=21,064,960/5,276,352; global instructions read/write=162,480/20,302.
- VL1 instructions read/write=162,400/20,332; L2C instructions read/write=332,823/82,463.
- Constant read path: total=48,832, SL1=29,876, L2=1,285.
- DNOC latency histogram total=152,962 samples, >512-cycle samples=33,674 (22.01%).
- VL1 partition stalls: min/avg/max=32,518/92,003/156,023 cycles, spread=134.24%.

DNOC latency buckets:

- 0-255: 48,202 samples (31.51%)
- 256-511: 71,086 samples (46.47%)
- 512-1k: 33,163 samples (21.68%)
- 1k-2k: 511 samples (0.33%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=162,480, write=20,302
- shared: read=0, write=0
- constant: read=48,832, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=162,400, write=20,332
- const_sl1c: read=29,876, write=0
- vl1c_l2c: read=332,823, write=82,463
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 2.26 TFLOP/s @ 727.91 GByte/s.
- Intensities: HBM=3.11, VL1=1.75, L2C=1.97 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=39.49%, VL1=8.63%, L2C=24.93%.
- Roofline raw context: all_ops=81,830,912, all_memacs=26,341,312, vl1_memacs=46,779,392, l2c_memacs=41,574,400, duration=40,711, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=5.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## 5. Inference Chain

1. Measured: MTE share=51.97% / AP MTE duty=41.75%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=5.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=51.97%, MMA share=0.00%, AP MTE duty=41.75%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=8.50%, L2C hit=44.82%, DNOC >512-cycle share=22.01%, DPC imbalance=7.06%, WIF P75/P25=1.36x, and effective occupancy=5.00%.
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
