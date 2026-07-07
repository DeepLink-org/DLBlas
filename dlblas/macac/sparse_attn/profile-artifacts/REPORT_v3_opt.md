# `sparse_attn_kernel_opt(unsigned short const*, unsigned short const*, int const*, float const*, unsigned short*, int, int, int, int, int, int, float)` Trace Profiling Report

**Tag:** `v3_opt`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v3_opt`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v3_opt/artifacts`

## 0. Profiling setup

- Kernel: `sparse_attn_kernel_opt(unsigned short const*, unsigned short const*, int const*, float const*, unsigned short*, int, int, int, int, int, int, float)`
- Grid / block: `[256, 1, 1]` / `[64, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v3_opt/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v3_opt/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run/profile-artifacts/sparse_attn_v3_opt/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag v3_opt \
  --cycle-dpc-id 2,3 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `compute`
- Primary: `compute`
- Confidence: High
- Dominant signal: MTE share=49.39% / AP MTE duty=11.77%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=14.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=49.39%, AP MTE duty=11.77%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,536,636 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 49.39% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 11.77% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 14.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 98.42% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 7.00x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 10.51% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 55.06% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 0.66% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 99.38% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[256, 1, 1]` / `[64, 1, 1]`; registers/thread=72, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=14.00% and shared-memory=0.00%; digest effective bound=14.00%.
- CycleTrace WIF mean/P25/P50/P75/max=12.18/3.00/10.00/21.00/32 raw waves; P75/P25=7.00x.
- mcProfiler achieved/dispatched waves=365/368, workgroups=284, average wave life=5,933 cycles.
- DPC compute balance: min/avg/max=185,468/202,824/206,784 cycles, imbalance=10.51%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 1,219 | 49.39% | Vector/Matrix Tensor Extension compute category |
| STE | 930 | 37.68% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 5 | 0.20% | Block shared-memory hardware category |
| GLOBAL | 146 | 5.92% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 142 | 5.75% | Synchronization/arrival category |
| LDU | 26 | 1.05% | Load data unit category |

- STE subevents by `name`: STE=623, S_NOP=216, Branch=91.
- GLOBAL subevents by `name`: GVM Load=145, GVM Store=1.
- ARRIVE subevents by `name`: Synchronization=142.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 1,219 / 49.39% | 327,607 / 54.95% | +5.55 pp |
| STE | 930 / 37.68% | 162,204 / 27.20% | -10.48 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 5 / 0.20% | 1,270 / 0.21% | +0.01 pp |
| GLOBAL/GVM | 146 / 5.92% | 37,191 / 6.24% | +0.32 pp |
| LDU | 26 / 1.05% | 7,144 / 1.20% | +0.14 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=11.77% / 5.58% / 0.00%.
- Real IPC=16.06; instruction throughput=0.1545; throughput efficiency=3.86%; compute-instruction busy duty=13.95%.
- Instructions per AP=5,733; average cycles/instruction=6.77; average all-stage latency/instruction=51.22 cycles.
- Compute instruction cycle split: total=2,637,120 cycles, MTE=2,637,120 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=590,524 cycles, top=vls_pipeline_stall 586,886 cycles (99.38%).
- AP active cycles=2,908,583; average AP busy cycles=28,239; AP busy duty=0.001081. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 914 cycles (0.15%)
- vls_pipeline_stall: 586,886 cycles (99.38%)
- vls_wdata_stall: 2,724 cycles (0.46%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=64.96% / 98.42% / 93.77%.
- DNOC read average latency=1,260 cycles; read/write requests=3,710/1,162; achieved bandwidth=12.17 GB/s.
- Global memory bytes read/write=258,304/143,168; global instructions read/write=36,830/361.
- VL1 instructions read/write=36,685/360; L2C instructions read/write=299,410/2,266.
- Constant read path: total=7,144, SL1=4,088, L2=264.
- DNOC latency histogram total=1,235 samples, >512-cycle samples=680 (55.06%).
- VL1 partition stalls: min/avg/max=154,119/172,723/191,011 cycles, spread=21.36%.

DNOC latency buckets:

- 0-255: 545 samples (44.13%)
- 256-511: 10 samples (0.81%)
- 512-1k: 299 samples (24.21%)
- 1k-2k: 0 samples (0.00%)
- 2k-: 381 samples (30.85%)

Memory data flow:

- global: read=36,830, write=361
- shared: read=1,012, write=253
- constant: read=7,144, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=36,685, write=360
- const_sl1c: read=4,088, write=0
- vl1c_l2c: read=299,410, write=2,266
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=1,012/253.
- Avg cycles per load/store=8.00/2.01; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=50.60/29.68/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.64 TFLOP/s @ 12.17 GByte/s.
- Intensities: HBM=52.57, VL1=6.56, L2C=0.59 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=0.66%, VL1=0.65%, L2C=23.54%.
- Roofline raw context: all_ops=21,106,496, all_memacs=401,472, vl1_memacs=3,216,968, l2c_memacs=35,784,320, duration=37,116, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=14.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## 5. Inference Chain

1. Measured: MTE share=49.39% / AP MTE duty=11.77%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=14.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=49.39%, MMA share=0.00%, AP MTE duty=11.77%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=5.92%, L2C hit=98.42%, DNOC >512-cycle share=55.06%, DPC imbalance=10.51%, WIF P75/P25=7.00x, and effective occupancy=14.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: focus the compute path indicated by SOL and compute-cycle split, reducing scalar/vector overhead or improving the intended compute instruction mix.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare AP duty ratios, CycleTrace instruction shares, compute instruction cycle split, shared memory efficiency, and `REPORT_<tag>.md`.
- Expected metric movement: the intended compute duty and compute-cycle share should increase without regressing memory hierarchy or occupancy metrics.

## 7. Artifacts

- Full metrics: `analysis/metrics_all_v3_opt.json`
- Key metrics: `analysis/metrics_key_v3_opt.json`
- Digest: `analysis/digest_v3_opt.md`
- This report: `REPORT_v3_opt.md`

## 8. Caveats

- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.
- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.
