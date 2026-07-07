# `indexer_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, int*, int, int, int, int, int, int, int, int)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/indexer_run/profile-artifacts/indexer_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/indexer_run/profile-artifacts/indexer_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `indexer_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, int*, int, int, int, int, int, int, int, int)`
- Grid / block: `[128, 1, 1]` / `[64, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/indexer_run/profile-artifacts/indexer_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/indexer_run/profile-artifacts/indexer_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/indexer_run/profile-artifacts/indexer_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=47.10% / AP MTE duty=3.40%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=5.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=47.10%, AP MTE duty=3.40%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,798,471 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 47.10% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 3.40% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 5.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 98.46% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 3.50x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 25.62% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 32.10% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 0.49% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | wsm_stall / 45.92% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[128, 1, 1]` / `[64, 1, 1]`; registers/thread=30, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=5.00% and shared-memory=0.00%; digest effective bound=5.00%.
- CycleTrace WIF mean/P25/P50/P75/max=9.33/4.00/8.00/14.00/24 raw waves; P75/P25=3.50x.
- mcProfiler achieved/dispatched waves=387/392, workgroups=194, average wave life=10,723 cycles.
- DPC compute balance: min/avg/max=881,852/1,039,926/1,148,312 cycles, imbalance=25.62%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 2,254 | 47.10% | Vector/Matrix Tensor Extension compute category |
| STE | 1,371 | 28.65% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 72 | 1.50% | Block shared-memory hardware category |
| GLOBAL | 528 | 11.03% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 557 | 11.64% | Synchronization/arrival category |
| LDU | 4 | 0.08% | Load data unit category |

- STE subevents by `name`: STE=765, S_NOP=433, Branch=173.
- GLOBAL subevents by `name`: GVM Load=512, GVM Store=16.
- ARRIVE subevents by `name`: Synchronization=557.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 2,254 / 47.10% | 332,580 / 54.13% | +7.03 pp |
| STE | 1,371 / 28.65% | 106,538 / 17.34% | -11.31 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 72 / 1.50% | 9,459 / 1.54% | +0.04 pp |
| GLOBAL/GVM | 528 / 11.03% | 67,388 / 10.97% | -0.06 pp |
| LDU | 4 / 0.08% | 1,824 / 0.30% | +0.21 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=3.40% / 1.09% / 0.00%.
- Real IPC=5.28; instruction throughput=0.05077; throughput efficiency=1.27%; compute-instruction busy duty=21.18%.
- Instructions per AP=5,908; average cycles/instruction=33.70; average all-stage latency/instruction=52.14 cycles.
- Compute instruction cycle split: total=1,612,296 cycles, MTE=1,612,296 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=635,261 cycles, top=wsm_stall 291,685 cycles (45.92%).
- AP active cycles=9,818,443; average AP busy cycles=95,325; AP busy duty=0.00339. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 291,685 cycles (45.92%)
- vls_pipeline_stall: 126,632 cycles (19.93%)
- vls_wdata_stall: 216,944 cycles (34.15%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=84.41% / 98.46% / 92.59%.
- DNOC read average latency=455.6 cycles; read/write requests=6,979/2,831; achieved bandwidth=9.06 GB/s.
- Global memory bytes read/write=579,360/357,952; global instructions read/write=64,512/2,368.
- VL1 instructions read/write=64,512/2,356; L2C instructions read/write=665,481/7,544.
- Constant read path: total=1,824, SL1=1,160, L2=92.
- DNOC latency histogram total=3,365 samples, >512-cycle samples=1,080 (32.10%).
- VL1 partition stalls: min/avg/max=100,406/127,252/188,430 cycles, spread=69.17%.

DNOC latency buckets:

- 0-255: 2,172 samples (64.55%)
- 256-511: 113 samples (3.36%)
- 512-1k: 272 samples (8.08%)
- 1k-2k: 12 samples (0.36%)
- 2k-: 796 samples (23.66%)

Memory data flow:

- global: read=64,512, write=2,368
- shared: read=8,128, write=128,025
- constant: read=1,824, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=64,512, write=2,356
- const_sl1c: read=1,160, write=0
- vl1c_l2c: read=665,481, write=7,544
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=8,128/128,025.
- Avg cycles per load/store=38.90/0.02; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=50.59/0.30/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.22 TFLOP/s @ 9.06 GByte/s.
- Intensities: HBM=24.67, VL1=1.38, L2C=0.28 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=0.49%, VL1=1.08%, L2C=17.32%.
- Roofline raw context: all_ops=23,127,744, all_memacs=937,312, vl1_memacs=16,741,312, l2c_memacs=82,575,360, duration=116,378, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

1. Measured: MTE share=47.10% / AP MTE duty=3.40%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=5.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=47.10%, MMA share=0.00%, AP MTE duty=3.40%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=11.03%, L2C hit=98.46%, DNOC >512-cycle share=32.10%, DPC imbalance=25.62%, WIF P75/P25=3.50x, and effective occupancy=5.00%.
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
