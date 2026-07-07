# `_Z20act_quant_kernel_optPKDF16_PDF16_Pfiiiff` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run/profile-artifacts/act_quant_kernel_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run/profile-artifacts/act_quant_kernel_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `_Z20act_quant_kernel_optPKDF16_PDF16_Pfiiiff`
- Grid / block: `[7, 1, 1]` / `[512, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run/profile-artifacts/act_quant_kernel_v0_baseline/artifacts/c-trace_output_dpc_0.json`

- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run/profile-artifacts/act_quant_kernel_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run/profile-artifacts/act_quant_kernel_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=37.36% / AP MTE duty=12.80%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=3.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=37.36%, AP MTE duty=12.80%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,700,848 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 37.36% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 12.80% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 3.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 25.52% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 2.86x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 76.58% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 40.49% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 0.82% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 68.39% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[7, 1, 1]` / `[512, 1, 1]`; registers/thread=24, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=4.00% and shared-memory=3.00%; digest effective bound=3.00%.
- CycleTrace WIF mean/P25/P50/P75/max=3.33/1.75/3.00/5.00/8 raw waves; P75/P25=2.86x.
- mcProfiler achieved/dispatched waves=88/88, workgroups=15, average wave life=2,211 cycles.
- DPC compute balance: min/avg/max=2,868/8,038/9,024 cycles, imbalance=76.58%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 272 | 37.36% | Vector/Matrix Tensor Extension compute category |
| STE | 347 | 47.66% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 7 | 0.96% | Block shared-memory hardware category |
| GLOBAL | 8 | 1.10% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 82 | 11.26% | Synchronization/arrival category |
| LDU | 12 | 1.65% | Load data unit category |

- STE subevents by `name`: STE=158, S_NOP=134, Branch=55.
- GLOBAL subevents by `name`: GVM Load=4, GVM Store=4.
- ARRIVE subevents by `name`: Synchronization=82.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 272 / 37.36% | 12,597 / 52.96% | +15.60 pp |
| STE | 347 / 47.66% | 5,605 / 23.56% | -24.10 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 7 / 0.96% | 385 / 1.62% | +0.66 pp |
| GLOBAL/GVM | 8 / 1.10% | 247 / 1.04% | -0.06 pp |
| LDU | 12 / 1.65% | 448 / 1.88% | +0.24 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=12.80% / 5.40% / 0.00%.
- Real IPC=3.11; instruction throughput=0.02989; throughput efficiency=0.75%; compute-instruction busy duty=15.48%.
- Instructions per AP=228.7; average cycles/instruction=1.71; average all-stage latency/instruction=39.04 cycles.
- Compute instruction cycle split: total=108,852 cycles, MTE=108,852 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=3,461 cycles, top=vls_pipeline_stall 2,367 cycles (68.39%).
- AP active cycles=103,824; average AP busy cycles=6,922; AP busy duty=0.000223. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 302 cycles (8.73%)
- vls_pipeline_stall: 2,367 cycles (68.39%)
- vls_wdata_stall: 792 cycles (22.88%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.47% / 25.52% / 95.75%.
- DNOC read average latency=385.8 cycles; read/write requests=1,087/274; achieved bandwidth=15.20 GB/s.
- Global memory bytes read/write=72,832/30,560; global instructions read/write=112/135.
- VL1 instructions read/write=112/131; L2C instructions read/write=1,540/553.
- Constant read path: total=448, SL1=476, L2=21.
- DNOC latency histogram total=1,808 samples, >512-cycle samples=732 (40.49%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 128 samples (7.08%)
- 256-511: 948 samples (52.43%)
- 512-1k: 5 samples (0.28%)
- 1k-2k: 727 samples (40.21%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=112, write=135
- shared: read=238, write=147
- constant: read=448, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=112, write=131
- const_sl1c: read=476, write=0
- vl1c_l2c: read=1,540, write=553
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=238/147.
- Avg cycles per load/store=1.71/1.71; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.61/30.06/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.12 TFLOP/s @ 15.20 GByte/s.
- Intensities: HBM=7.78, VL1=12.82, L2C=56.13 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=0.82%, VL1=0.06%, L2C=0.05%.
- Roofline raw context: all_ops=804,672, all_memacs=103,392, vl1_memacs=62,776, l2c_memacs=14,336, duration=7,652, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: NOP share=18.41%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=37.36% / AP MTE duty=12.80%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=3.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=37.36%, MMA share=0.00%, AP MTE duty=12.80%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=1.10%, L2C hit=25.52%, DNOC >512-cycle share=40.49%, DPC imbalance=76.58%, WIF P75/P25=2.86x, and effective occupancy=3.00%.
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
