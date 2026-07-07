# `engram_gate_bwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, float*, float*, float*, int, int, int, float, float)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/mnt/opt_test/engram_gate_bwd_run/profile-artifacts/engram_gate_bwd_v0_baseline`
**Artifact directory:** `/mnt/opt_test/engram_gate_bwd_run/profile-artifacts/engram_gate_bwd_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `engram_gate_bwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, unsigned short*, float*, float*, float*, int, int, int, float, float)`
- Grid / block: `[56, 1, 1]` / `[128, 1, 1]`
- CycleTrace JSON: `/mnt/opt_test/engram_gate_bwd_run/profile-artifacts/engram_gate_bwd_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/mnt/opt_test/engram_gate_bwd_run/profile-artifacts/engram_gate_bwd_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/mnt/opt_test/engram_gate_bwd_run/profile-artifacts/engram_gate_bwd_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=50.18% / AP MTE duty=8.19%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=2.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=50.18%, AP MTE duty=8.19%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 1,869,755 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 50.18% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 8.19% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 2.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 50.05% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 5.25x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 9.73% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 8.62% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 3.37% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | wsm_stall / 37.69% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[56, 1, 1]` / `[128, 1, 1]`; registers/thread=26, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=5.00% and shared-memory=2.00%; digest effective bound=2.00%.
- CycleTrace WIF mean/P25/P50/P75/max=4.06/1.00/3.00/5.25/14 raw waves; P75/P25=5.25x.
- mcProfiler achieved/dispatched waves=266/268, workgroups=95, average wave life=2,859 cycles.
- DPC compute balance: min/avg/max=38,188/40,779/42,156 cycles, imbalance=9.73%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 275 | 50.18% | Vector/Matrix Tensor Extension compute category |
| STE | 165 | 30.11% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 8 | 1.46% | Block shared-memory hardware category |
| GLOBAL | 21 | 3.83% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 71 | 12.96% | Synchronization/arrival category |
| LDU | 8 | 1.46% | Load data unit category |

- STE subevents by `name`: STE=78, S_NOP=67, Branch=20.
- GLOBAL subevents by `name`: GVM Load=16, GVM Store=2.
- ARRIVE subevents by `name`: Synchronization=71.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 275 / 50.18% | 58,518 / 61.01% | +10.83 pp |
| STE | 165 / 30.11% | 14,142 / 14.74% | -15.37 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 8 / 1.46% | 5,500 / 5.73% | +4.27 pp |
| GLOBAL/GVM | 21 / 3.83% | 2,436 / 2.54% | -1.29 pp |
| LDU | 8 / 1.46% | 1,660 / 1.73% | +0.27 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=8.19% / 1.83% / 0.00%.
- Real IPC=10.13; instruction throughput=0.09741; throughput efficiency=2.44%; compute-instruction busy duty=10.58%.
- Instructions per AP=922.2; average cycles/instruction=2.04; average all-stage latency/instruction=38.07 cycles.
- Compute instruction cycle split: total=621,280 cycles, MTE=621,280 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=12,820 cycles, top=wsm_stall 4,832 cycles (37.69%).
- AP active cycles=770,800; average AP busy cycles=8,200; AP busy duty=0.0002759. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 4,832 cycles (37.69%)
- vls_pipeline_stall: 3,772 cycles (29.42%)
- vls_wdata_stall: 4,216 cycles (32.89%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.51% / 50.05% / 91.66%.
- DNOC read average latency=1,031 cycles; read/write requests=5,113/1,443; achieved bandwidth=62.14 GB/s.
- Global memory bytes read/write=350,208/172,800; global instructions read/write=1,760/346.
- VL1 instructions read/write=1,792/348; L2C instructions read/write=10,347/2,583.
- Constant read path: total=1,660, SL1=1,032, L2=86.
- DNOC latency histogram total=5,177 samples, >512-cycle samples=446 (8.62%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 688 samples (13.29%)
- 256-511: 4,043 samples (78.10%)
- 512-1k: 446 samples (8.62%)
- 1k-2k: 0 samples (0.00%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=1,760, write=346
- shared: read=3,584, write=1,980
- constant: read=1,660, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=1,792, write=348
- const_sl1c: read=1,032, write=0
- vl1c_l2c: read=10,347, write=2,583
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=3,584/1,980.
- Avg cycles per load/store=2.00/2.04; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.59/30.18/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 2,346 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.45 TFLOP/s @ 62.14 GByte/s.
- Intensities: HBM=7.17, VL1=5.92, L2C=16.34 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=3.37%, VL1=0.50%, L2C=0.59%.
- Roofline raw context: all_ops=3,748,608, all_memacs=523,008, vl1_memacs=633,216, l2c_memacs=229,376, duration=9,468, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=5.00%, smem=2.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=12.23%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=50.18% / AP MTE duty=8.19%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=2.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=50.18%, MMA share=0.00%, AP MTE duty=8.19%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=3.83%, L2C hit=50.05%, DNOC >512-cycle share=8.62%, DPC imbalance=9.73%, WIF P75/P25=5.25x, and effective occupancy=2.00%.
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
