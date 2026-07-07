# `mhc_post_kernel_opt(__maca_bfloat16 const*, __maca_bfloat16 const*, float const*, float const*, __maca_bfloat16*, int, int, int, int)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/opt_test/mhc_post_run/profile-artifacts/mhc_post_v0_baseline`
**Artifact directory:** `/home/ailab/opt_test/mhc_post_run/profile-artifacts/mhc_post_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `mhc_post_kernel_opt(__maca_bfloat16 const*, __maca_bfloat16 const*, float const*, float const*, __maca_bfloat16*, int, int, int, int)`
- Grid / block: `[832, 1, 1]` / `[512, 1, 1]`
- CycleTrace JSON: `/home/ailab/opt_test/mhc_post_run/profile-artifacts/mhc_post_v0_baseline/artifacts/c-trace_output_dpc_0.json`

- mcTracer JSON: `/home/ailab/opt_test/mhc_post_run/profile-artifacts/mhc_post_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/opt_test/mhc_post_run/profile-artifacts/mhc_post_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=62.60% / AP MTE duty=69.15%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=8.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=62.60%, AP MTE duty=69.15%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,068,811,235 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 62.60% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 69.15% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 8.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 0.58% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 1.97x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.80% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 53.45% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 66.86% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 64.67% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[832, 1, 1]` / `[512, 1, 1]`; registers/thread=42, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=8.00% and shared-memory=0.00%; digest effective bound=8.00%.
- CycleTrace WIF mean/P25/P50/P75/max=266.66/181.00/318.00/357.00/416 raw waves; P75/P25=1.97x.
- mcProfiler achieved/dispatched waves=6,896/6,964, workgroups=909, average wave life=16,043 cycles.
- DPC compute balance: min/avg/max=5,541,664/5,945,722/6,005,168 cycles, imbalance=7.80%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 28,776 | 62.60% | Vector/Matrix Tensor Extension compute category |
| STE | 10,653 | 23.17% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 3,555 | 7.73% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 2,434 | 5.29% | Synchronization/arrival category |
| LDU | 551 | 1.20% | Load data unit category |

- STE subevents by `name`: STE=7,912, S_NOP=2,156, Branch=585.
- GLOBAL subevents by `name`: GVM Load=1,975, GVM Store=1,580.
- ARRIVE subevents by `name`: Synchronization=2,434.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 28,776 / 62.60% | 11,883,764 / 65.65% | +3.05 pp |
| STE | 10,653 / 23.17% | 3,276,188 / 18.10% | -5.08 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 3,555 / 7.73% | 1,460,976 / 8.07% | +0.34 pp |
| LDU | 551 / 1.20% | 229,192 / 1.27% | +0.07 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=69.15% / 19.06% / 0.00%.
- Real IPC=104.14; instruction throughput=1.001; throughput efficiency=25.03%; compute-instruction busy duty=69.19%.
- Instructions per AP=174,051; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=47,682,512 cycles, MTE=47,682,512 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=7,377,417 cycles, top=vls_pipeline_stall 4,771,081 cycles (64.67%).
- AP active cycles=17,186,582; average AP busy cycles=166,860; AP busy duty=0.003852. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 4,771,081 cycles (64.67%)
- vls_wdata_stall: 2,606,336 cycles (35.33%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=97.73% / 0.58% / 92.81%.
- DNOC read average latency=548.1 cycles; read/write requests=834,668/660,511; achieved bandwidth=1232.36 GB/s.
- Global memory bytes read/write=105,854,912/84,542,400; global instructions read/write=811,400/649,676.
- VL1 instructions read/write=811,300/649,668; L2C instructions read/write=843,746/1,320,989.
- Constant read path: total=229,192, SL1=140,152, L2=10,037.
- DNOC latency histogram total=833,482 samples, >512-cycle samples=445,460 (53.45%).
- VL1 partition stalls: min/avg/max=0/83.25/315 cycles, spread=378.38%.

DNOC latency buckets:

- 0-255: 97,469 samples (11.69%)
- 256-511: 290,553 samples (34.86%)
- 512-1k: 420,924 samples (50.50%)
- 1k-2k: 24,534 samples (2.94%)
- 2k-: 2 samples (0.00%)

Memory data flow:

- global: read=811,400, write=649,676
- shared: read=0, write=0
- constant: read=229,192, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=811,300, write=649,668
- const_sl1c: read=140,152, write=0
- vl1c_l2c: read=843,746, write=1,320,989
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 6.00 TFLOP/s @ 1232.36 GByte/s.
- Intensities: HBM=4.87, VL1=2.48, L2C=8.92 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=66.86%, VL1=16.16%, L2C=14.59%.
- Roofline raw context: all_ops=926,598,400, all_memacs=190,397,312, vl1_memacs=374,009,856, l2c_memacs=103,859,200, duration=173,811, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=8.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## 5. Inference Chain

1. Measured: MTE share=62.60% / AP MTE duty=69.15%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=8.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=62.60%, MMA share=0.00%, AP MTE duty=69.15%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=7.73%, L2C hit=0.58%, DNOC >512-cycle share=53.45%, DPC imbalance=7.80%, WIF P75/P25=1.97x, and effective occupancy=8.00%.
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
