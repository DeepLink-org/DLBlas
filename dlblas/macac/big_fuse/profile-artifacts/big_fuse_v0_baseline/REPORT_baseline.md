# `_Z19big_fuse_kernel_optPKDF16_PKfS2_S2_PfS3_PDF16_iiiiiffffi` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v0_baseline`
**Artifact directory:** `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `_Z19big_fuse_kernel_optPKDF16_PKfS2_S2_PfS3_PDF16_iiiiiffffi`
- Grid / block: `[512, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Confidence: High
- Dominant signal: MTE share=49.44% / AP MTE duty=34.28%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=49.44%, AP MTE duty=34.28%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 3,524,486 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 49.44% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 34.28% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 1.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 96.77% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 4.95x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 6.31% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 8.35% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 0.82% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 78.70% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[512, 1, 1]` / `[256, 1, 1]`; registers/thread=108, static shared=1,024 bytes.
- mcTracer occupancy fields are mtreg=21.00% and shared-memory=1.00%; digest effective bound=1.00%.
- CycleTrace WIF mean/P25/P50/P75/max=108.37/37.00/110.00/183.00/208 raw waves; P75/P25=4.95x.
- mcProfiler achieved/dispatched waves=2,300/2,316, workgroups=579, average wave life=76,493 cycles.
- DPC compute balance: min/avg/max=13,857,664/14,664,905/14,782,576 cycles, imbalance=6.31%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 45,075 | 49.44% | Vector/Matrix Tensor Extension compute category |
| STE | 36,326 | 39.84% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 600 | 0.66% | Block shared-memory hardware category |
| GLOBAL | 2,625 | 2.88% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 6,490 | 7.12% | Synchronization/arrival category |
| LDU | 60 | 0.07% | Load data unit category |

- STE subevents by `name`: STE=21,280, S_NOP=11,976, Branch=3,070.
- GLOBAL subevents by `name`: GVM Load=2,600, GVM Store=25.
- ARRIVE subevents by `name`: Synchronization=6,490.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 45,075 / 49.44% | 18,494,544 / 56.50% | +7.06 pp |
| STE | 36,326 / 39.84% | 8,704,133 / 26.59% | -13.25 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 600 / 0.66% | 426,720 / 1.30% | +0.65 pp |
| GLOBAL/GVM | 2,625 / 2.88% | 1,066,276 / 3.26% | +0.38 pp |
| LDU | 60 / 0.07% | 25,656 / 0.08% | +0.01 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=34.28% / 14.42% / 0.00%.
- Real IPC=53.19; instruction throughput=0.5114; throughput efficiency=12.78%; compute-instruction busy duty=48.60%.
- Instructions per AP=314,735; average cycles/instruction=2.00; average all-stage latency/instruction=39.04 cycles.
- Compute instruction cycle split: total=255,168,584 cycles, MTE=255,168,584 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=2,032,198 cycles, top=vls_pipeline_stall 1,599,350 cycles (78.70%).
- AP active cycles=60,353,784; average AP busy cycles=585,959; AP busy duty=0.01759. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 379,444 cycles (18.67%)
- vls_pipeline_stall: 1,599,350 cycles (78.70%)
- vls_wdata_stall: 53,404 cycles (2.63%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.98% / 96.77% / 98.58%.
- DNOC read average latency=144.5 cycles; read/write requests=52,467/15,103; achieved bandwidth=15.08 GB/s.
- Global memory bytes read/write=6,323,360/1,927,264; global instructions read/write=1,054,560/11,738.
- VL1 instructions read/write=1,054,560/11,738; L2C instructions read/write=2,071,408/30,528.
- Constant read path: total=25,656, SL1=15,784, L2=222.
- DNOC latency histogram total=49,219 samples, >512-cycle samples=4,109 (8.35%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 42,711 samples (86.78%)
- 256-511: 2,399 samples (4.87%)
- 512-1k: 3,649 samples (7.41%)
- 1k-2k: 102 samples (0.21%)
- 2k-: 358 samples (0.73%)

Memory data flow:

- global: read=1,054,560, write=11,738
- shared: read=267,696, write=158,184
- constant: read=25,656, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=1,054,560, write=11,738
- const_sl1c: read=15,784, write=0
- vl1c_l2c: read=2,071,408, write=30,528
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=267,696/158,184.
- Avg cycles per load/store=2.00/2.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.66/30.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 2.52 TFLOP/s @ 15.08 GByte/s.
- Intensities: HBM=167.09, VL1=5.05, L2C=5.30 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=0.82%, VL1=3.33%, L2C=10.32%.
- Roofline raw context: all_ops=1,378,619,904, all_memacs=8,250,624, vl1_memacs=272,747,156, l2c_memacs=260,096,000, duration=615,443, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=21.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=13.13%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=49.44% / AP MTE duty=34.28%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=49.44%, MMA share=0.00%, AP MTE duty=34.28%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=2.88%, L2C hit=96.77%, DNOC >512-cycle share=8.35%, DPC imbalance=6.31%, WIF P75/P25=4.95x, and effective occupancy=1.00%.
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
