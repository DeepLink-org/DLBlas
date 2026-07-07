# `norm_fn_kernel_opt(float const*, float const*, float*, int, int, int, float)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v0_baseline`
**Artifact directory:** `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `norm_fn_kernel_opt(float const*, float const*, float*, int, int, int, float)`
- Grid / block: `[24, 13, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/opt_test/norm_fn_run/profile-artifacts/norm_fn_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=36.97% / AP MTE duty=16.00%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=36.97%, AP MTE duty=16.00%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 2,039,819 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 36.97% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 16.00% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 1.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 83.85% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 6.56x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 6.11% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 3.58% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 3.80% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 70.97% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[24, 13, 1]` / `[256, 1, 1]`; registers/thread=14, static shared=1,024 bytes.
- mcTracer occupancy fields are mtreg=2.00% and shared-memory=1.00%; digest effective bound=1.00%.
- CycleTrace WIF mean/P25/P50/P75/max=62.08/16.00/54.00/105.00/156 raw waves; P75/P25=6.56x.
- mcProfiler achieved/dispatched waves=1,600/1,616, workgroups=404, average wave life=5,381 cycles.
- DPC compute balance: min/avg/max=223,188/233,362/237,448 cycles, imbalance=6.11%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 858 | 36.97% | Vector/Matrix Tensor Extension compute category |
| STE | 1,025 | 44.16% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 9 | 0.39% | Block shared-memory hardware category |
| GLOBAL | 120 | 5.17% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 294 | 12.67% | Synchronization/arrival category |
| LDU | 15 | 0.65% | Load data unit category |

- STE subevents by `name`: STE=432, S_NOP=419, Branch=174.
- GLOBAL subevents by `name`: GVM Load=120, GVM Store=0.
- ARRIVE subevents by `name`: Synchronization=294.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 858 / 36.97% | 445,523 / 47.87% | +10.90 pp |
| STE | 1,025 / 44.16% | 195,414 / 20.99% | -23.17 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 9 / 0.39% | 20,703 / 2.22% | +1.84 pp |
| GLOBAL/GVM | 120 / 5.17% | 50,491 / 5.42% | +0.25 pp |
| LDU | 15 / 0.65% | 8,016 / 0.86% | +0.21 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=16.00% / 6.96% / 0.00%.
- Real IPC=29.84; instruction throughput=0.2869; throughput efficiency=7.17%; compute-instruction busy duty=16.62%.
- Instructions per AP=8,950; average cycles/instruction=2.00; average all-stage latency/instruction=39.02 cycles.
- Compute instruction cycle split: total=2,124,196 cycles, MTE=2,124,196 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=106,716 cycles, top=vls_pipeline_stall 75,737 cycles (70.97%).
- AP active cycles=2,808,861; average AP busy cycles=27,270; AP busy duty=0.0009088. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 17,871 cycles (16.75%)
- vls_pipeline_stall: 75,737 cycles (70.97%)
- vls_wdata_stall: 13,108 cycles (12.28%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.55% / 83.85% / 98.02%.
- DNOC read average latency=878.7 cycles; read/write requests=12,610/5,993; achieved bandwidth=70.02 GB/s.
- Global memory bytes read/write=1,178,656/763,008; global instructions read/write=49,440/1,047.
- VL1 instructions read/write=49,440/1,051; L2C instructions read/write=109,171/12,244.
- Constant read path: total=8,016, SL1=5,000, L2=98.
- DNOC latency histogram total=7,244 samples, >512-cycle samples=259 (3.58%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 5,970 samples (82.41%)
- 256-511: 1,015 samples (14.01%)
- 512-1k: 191 samples (2.64%)
- 1k-2k: 68 samples (0.94%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=49,440, write=1,047
- shared: read=12,669, write=8,034
- constant: read=8,016, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=49,440, write=1,051
- const_sl1c: read=5,000, write=0
- vl1c_l2c: read=109,171, write=12,244
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=12,669/8,034.
- Avg cycles per load/store=2.00/2.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.53/30.04/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 1.14 TFLOP/s @ 70.02 GByte/s.
- Intensities: HBM=16.24, VL1=2.45, L2C=2.49 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=3.80%, VL1=3.09%, L2C=9.90%.
- Roofline raw context: all_ops=31,534,784, all_memacs=1,941,664, vl1_memacs=12,846,916, l2c_memacs=12,656,640, duration=31,197, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=2.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=18.05%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=36.97% / AP MTE duty=16.00%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=36.97%, MMA share=0.00%, AP MTE duty=16.00%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=5.17%, L2C hit=83.85%, DNOC >512-cycle share=3.58%, DPC imbalance=6.11%, WIF P75/P25=6.56x, and effective occupancy=1.00%.
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
