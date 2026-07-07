# `expand_kenel_fwd_kernel_opt(float const*, float*, int, int, int, int)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/expand_kenel_fwd_run/profile-artifacts/expand_kenel_fwd_v0_baseline`
**Artifact directory:** `/mnt/opt_test/expand_kenel_fwd_run/profile-artifacts/expand_kenel_fwd_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `expand_kenel_fwd_kernel_opt(float const*, float*, int, int, int, int)`
- Grid / block: `[1024, 1, 1]` / `[320, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/expand_kenel_fwd_run/profile-artifacts/expand_kenel_fwd_v0_baseline/artifacts/c-trace_output_dpc_0.json`

- mcTracer JSON: `/mnt/opt_test/expand_kenel_fwd_run/profile-artifacts/expand_kenel_fwd_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/expand_kenel_fwd_run/profile-artifacts/expand_kenel_fwd_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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

- Bottleneck class: `compute`
- Primary: `compute`
- Confidence: Medium
- Dominant signal: MTE share=37.37% / AP MTE duty=8.32%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=2.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=37.37%, AP MTE duty=8.32%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 23,801 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 37.37% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 8.32% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 2.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 0.72% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 2.39x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.89% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 0.00% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 92.74% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 54.68% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[1024, 1, 1]` / `[320, 1, 1]`; registers/thread=12, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=2.00% and shared-memory=0.00%; digest effective bound=2.00%.
- CycleTrace WIF mean/P25/P50/P75/max=268.30/160.00/320.00/382.00/392 raw waves; P75/P25=2.39x.
- mcProfiler achieved/dispatched waves=5,075/5,120, workgroups=1,024, average wave life=2,689 cycles.
- DPC compute balance: min/avg/max=66,080/70,980/71,680 cycles, imbalance=7.89%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 364 | 37.37% | Vector/Matrix Tensor Extension compute category |
| STE | 441 | 45.28% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 65 | 6.67% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 65 | 6.67% | Synchronization/arrival category |
| LDU | 39 | 4.00% | Load data unit category |

- STE subevents by `name`: STE=351, S_NOP=77, Branch=13.
- GLOBAL subevents by `name`: GVM Load=13, GVM Store=52.
- ARRIVE subevents by `name`: Synchronization=65.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 364 / 37.37% | 142,100 / 40.02% | +2.64 pp |
| STE | 441 / 45.28% | 136,890 / 38.55% | -6.73 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 65 / 6.67% | 25,375 / 7.15% | +0.47 pp |
| LDU | 39 / 4.00% | 15,225 / 4.29% | +0.28 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=8.32% / 8.02% / 0.00%.
- Real IPC=20.58; instruction throughput=0.1979; throughput efficiency=4.95%; compute-instruction busy duty=8.32%.
- Instructions per AP=3,415; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=568,400 cycles, MTE=568,400 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=716,638 cycles, top=vls_pipeline_stall 391,838 cycles (54.68%).
- AP active cycles=1,706,931; average AP busy cycles=16,572; AP busy duty=0.0005023. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 391,838 cycles (54.68%)
- vls_wdata_stall: 324,800 cycles (45.32%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=77.50% / 0.72% / 99.66%.
- DNOC read average latency=388.5 cycles; read/write requests=41,000/163,861; achieved bandwidth=1709.33 GB/s.
- Global memory bytes read/write=5,245,152/20,972,160; global instructions read/write=5,075/20,300.
- VL1 instructions read/write=5,075/20,300; L2C instructions read/write=42,479/327,692.
- Constant read path: total=15,225, SL1=9,360, L2=32.
- DNOC latency histogram total=40,978 samples, >512-cycle samples=1 (0.00%).
- VL1 partition stalls: min/avg/max=6/1,832/3,698 cycles, spread=201.53%.

DNOC latency buckets:

- 0-255: 40,343 samples (98.45%)
- 256-511: 634 samples (1.55%)
- 512-1k: 1 samples (0.00%)
- 1k-2k: 0 samples (0.00%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=5,075, write=20,300
- shared: read=0, write=0
- constant: read=15,225, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=5,075, write=20,300
- const_sl1c: read=9,360, write=0
- vl1c_l2c: read=42,479, write=327,692
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.59 TFLOP/s @ 1709.33 GByte/s.
- Intensities: HBM=0.35, VL1=1.40, L2C=1.75 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=92.74%, VL1=2.83%, L2C=7.35%.
- Roofline raw context: all_ops=9,094,400, all_memacs=26,217,312, vl1_memacs=6,496,000, l2c_memacs=5,196,800, duration=17,255, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

1. Measured: MTE share=37.37% / AP MTE duty=8.32%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=2.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=37.37%, MMA share=0.00%, AP MTE duty=8.32%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=6.67%, L2C hit=0.72%, DNOC >512-cycle share=0.00%, DPC imbalance=7.89%, WIF P75/P25=2.39x, and effective occupancy=2.00%.
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
