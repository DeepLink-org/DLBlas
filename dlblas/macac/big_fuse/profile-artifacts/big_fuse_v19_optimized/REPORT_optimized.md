# `_Z19big_fuse_kernel_optPKDF16_PKfS2_S2_PfS3_PDF16_iiiiiffffi` Trace Profiling Report

**Tag:** `optimized`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v19_optimized`
**Artifact directory:** `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v19_optimized/artifacts`

## 0. Profiling setup

- Kernel: `_Z19big_fuse_kernel_optPKDF16_PKfS2_S2_PfS3_PDF16_iiiiiffffi`
- Grid / block: `[512, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v19_optimized/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v19_optimized/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/big_fuse_run/profile-artifacts/big_fuse_v19_optimized/artifacts/mcprofiler_report_dumped.json`
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
- Confidence: High
- Dominant signal: MTE share=49.58% / AP MTE duty=55.13%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=49.58%, AP MTE duty=55.13%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 3,323,867 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 49.58% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 55.13% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 1.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 96.71% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 4.60x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.89% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 10.33% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 1.52% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 72.03% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[512, 1, 1]` / `[256, 1, 1]`; registers/thread=50, static shared=1,024 bytes.
- mcTracer occupancy fields are mtreg=9.00% and shared-memory=1.00%; digest effective bound=1.00%.
- CycleTrace WIF mean/P25/P50/P75/max=114.56/40.00/112.00/184.00/256 raw waves; P75/P25=4.60x.
- mcProfiler achieved/dispatched waves=2,292/2,316, workgroups=579, average wave life=79,047 cycles.
- DPC compute balance: min/avg/max=12,696,188/13,636,040/13,772,656 cycles, imbalance=7.89%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 40,025 | 49.58% | Vector/Matrix Tensor Extension compute category |
| STE | 33,804 | 41.88% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 780 | 0.97% | Block shared-memory hardware category |
| GLOBAL | 2,625 | 3.25% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 3,430 | 4.25% | Synchronization/arrival category |
| LDU | 60 | 0.07% | Load data unit category |

- STE subevents by `name`: STE=23,238, S_NOP=9,486, Branch=1,080.
- GLOBAL subevents by `name`: GVM Load=2,600, GVM Store=25.
- ARRIVE subevents by `name`: Synchronization=3,430.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 40,025 / 49.58% | 16,527,138 / 55.72% | +6.14 pp |
| STE | 33,804 / 41.88% | 9,777,578 / 32.97% | -8.91 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 780 / 0.97% | 355,914 / 1.20% | +0.23 pp |
| GLOBAL/GVM | 2,625 / 3.25% | 1,066,264 / 3.60% | +0.34 pp |
| LDU | 60 / 0.07% | 25,676 / 0.09% | +0.01 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=55.13% / 28.90% / 0.00%.
- Real IPC=87.46; instruction throughput=0.841; throughput efficiency=21.02%; compute-instruction busy duty=80.60%.
- Instructions per AP=285,188; average cycles/instruction=2.46; average all-stage latency/instruction=53.77 cycles.
- Compute instruction cycle split: total=247,030,340 cycles, MTE=247,030,340 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=2,220,568 cycles, top=vls_pipeline_stall 1,599,512 cycles (72.03%).
- AP active cycles=33,837,863; average AP busy cycles=328,523; AP busy duty=0.009693. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 567,652 cycles (25.56%)
- vls_pipeline_stall: 1,599,512 cycles (72.03%)
- vls_wdata_stall: 53,404 cycles (2.40%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.98% / 96.71% / 98.61%.
- DNOC read average latency=141.5 cycles; read/write requests=54,302/15,112; achieved bandwidth=28.10 GB/s.
- Global memory bytes read/write=6,544,032/1,927,232; global instructions read/write=1,054,560/11,712.
- VL1 instructions read/write=1,054,560/11,708; L2C instructions read/write=2,080,420/30,528.
- Constant read path: total=25,676, SL1=15,884, L2=216.
- DNOC latency histogram total=56,209 samples, >512-cycle samples=5,805 (10.33%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 48,514 samples (86.31%)
- 256-511: 1,890 samples (3.36%)
- 512-1k: 581 samples (1.03%)
- 1k-2k: 3,406 samples (6.06%)
- 2k-: 1,818 samples (3.23%)

Memory data flow:

- global: read=1,054,560, write=11,712
- shared: read=24,336, write=15,210
- constant: read=25,676, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=1,054,560, write=11,708
- const_sl1c: read=15,884, write=0
- vl1c_l2c: read=2,080,420, write=30,528
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=24,336/15,210.
- Avg cycles per load/store=31.00/8.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=45.44/45.88/43.32 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 632,779 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 4.16 TFLOP/s @ 28.10 GByte/s.
- Intensities: HBM=147.88, VL1=4.59, L2C=4.83 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=1.52%, VL1=6.04%, L2C=18.69%.
- Roofline raw context: all_ops=1,252,707,456, all_memacs=8,471,264, vl1_memacs=272,747,156, l2c_memacs=259,584,000, duration=339,117, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=9.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=11.75%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=49.58% / AP MTE duty=55.13%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=49.58%, MMA share=0.00%, AP MTE duty=55.13%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=3.25%, L2C hit=96.71%, DNOC >512-cycle share=10.33%, DPC imbalance=7.89%, WIF P75/P25=4.60x, and effective occupancy=1.00%.
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
