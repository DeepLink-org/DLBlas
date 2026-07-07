# `_Z30engram_fused_weight_kernel_optPKDF16_S0_Pfi` Trace Profiling Report

**Tag:** `optimized`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/engram_fused_weight_run/profile-artifacts/engram_fused_weight_v1_optimized`
**Artifact directory:** `/mnt/opt_test/engram_fused_weight_run/profile-artifacts/engram_fused_weight_v1_optimized/artifacts`

## 0. Profiling setup

- Kernel: `_Z30engram_fused_weight_kernel_optPKDF16_S0_Pfi`
- Grid / block: `[2, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/engram_fused_weight_run/profile-artifacts/engram_fused_weight_v1_optimized/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/mnt/opt_test/engram_fused_weight_run/profile-artifacts/engram_fused_weight_v1_optimized/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/engram_fused_weight_run/profile-artifacts/engram_fused_weight_v1_optimized/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag optimized \
  --cycle-dpc-id 2,3 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `compute`
- Primary: `compute`
- Confidence: Medium
- Dominant signal: MTE share=36.92% / AP MTE duty=5.25%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=1.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=36.92%, AP MTE duty=5.25%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 4,149,731 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 36.92% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 5.25% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 1.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 29.75% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 3.00x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 33.77% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 17.11% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 3.84% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 52.28% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[2, 1, 1]` / `[256, 1, 1]`; registers/thread=6, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=1.00% and shared-memory=0.00%; digest effective bound=1.00%.
- CycleTrace WIF mean/P25/P50/P75/max=2.00/1.00/2.00/3.00/4 raw waves; P75/P25=3.00x.
- mcProfiler achieved/dispatched waves=848/856, workgroups=214, average wave life=605.4 cycles.
- DPC compute balance: min/avg/max=9,216/11,253/13,016 cycles, imbalance=33.77%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 48 | 36.92% | Vector/Matrix Tensor Extension compute category |
| STE | 54 | 41.54% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 6 | 4.62% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 14 | 10.77% | Synchronization/arrival category |
| LDU | 8 | 6.15% | Load data unit category |

- STE subevents by `name`: STE=36, S_NOP=16, Branch=2.
- GLOBAL subevents by `name`: GVM Load=4, GVM Store=2.
- ARRIVE subevents by `name`: Synchronization=14.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 48 / 36.92% | 22,026 / 43.09% | +6.16 pp |
| STE | 54 / 41.54% | 15,498 / 30.32% | -11.22 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 6 / 4.62% | 2,502 / 4.89% | +0.28 pp |
| LDU | 8 / 6.15% | 3,408 / 6.67% | +0.51 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=5.25% / 3.68% / 0.00%.
- Real IPC=4.64; instruction throughput=0.04462; throughput efficiency=1.12%; compute-instruction busy duty=5.34%.
- Instructions per AP=491.5; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=96,168 cycles, MTE=96,168 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=7,175 cycles, top=vls_pipeline_stall 3,751 cycles (52.28%).
- AP active cycles=421,591; average AP busy cycles=4,093; AP busy duty=0.0003206. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 3,751 cycles (52.28%)
- vls_wdata_stall: 3,424 cycles (47.72%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.82% / 29.75% / 93.73%.
- DNOC read average latency=155.1 cycles; read/write requests=5,427/2,915; achieved bandwidth=70.86 GB/s.
- Global memory bytes read/write=440,896/253,056; global instructions read/write=1,664/838.
- VL1 instructions read/write=1,664/838; L2C instructions read/write=8,332/4,707.
- Constant read path: total=3,408, SL1=2,088, L2=132.
- DNOC latency histogram total=6,300 samples, >512-cycle samples=1,078 (17.11%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 4,331 samples (68.75%)
- 256-511: 891 samples (14.14%)
- 512-1k: 50 samples (0.79%)
- 1k-2k: 1,028 samples (16.32%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=1,664, write=838
- shared: read=0, write=0
- constant: read=3,408, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=1,664, write=838
- const_sl1c: read=2,088, write=0
- vl1c_l2c: read=8,332, write=4,707
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.14 TFLOP/s @ 70.86 GByte/s.
- Intensities: HBM=2.02, VL1=2.19, L2C=6.59 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=3.84%, VL1=0.44%, L2C=0.47%.
- Roofline raw context: all_ops=1,403,520, all_memacs=693,952, vl1_memacs=640,512, l2c_memacs=212,992, duration=11,017, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: NOP share=12.31%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=36.92% / AP MTE duty=5.25%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=1.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=36.92%, MMA share=0.00%, AP MTE duty=5.25%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=4.62%, L2C hit=29.75%, DNOC >512-cycle share=17.11%, DPC imbalance=33.77%, WIF P75/P25=3.00x, and effective occupancy=1.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: focus the compute path indicated by SOL and compute-cycle split, reducing scalar/vector overhead or improving the intended compute instruction mix.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare AP duty ratios, CycleTrace instruction shares, compute instruction cycle split, shared memory efficiency, and `REPORT_<tag>.md`.
- Expected metric movement: the intended compute duty and compute-cycle share should increase without regressing memory hierarchy or occupancy metrics.

## 7. Artifacts

- Full metrics: `analysis/metrics_all_optimized.json`
- Key metrics: `analysis/metrics_key_optimized.json`
- Digest: `analysis/digest_optimized.md`
- This report: `REPORT_optimized.md`

## 8. Caveats

- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.
- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.
