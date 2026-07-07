# `engram_gate_fwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, float*, float*, float*, float*, int, int, int, float, float, float)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `engram_gate_fwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, float*, float*, float*, float*, int, int, int, float, float, float)`
- Grid / block: `[16384, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=54.12% / AP MTE duty=65.44%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=4.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=54.12%, AP MTE duty=65.44%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 442,893,679 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 54.12% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 65.44% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 4.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 39.64% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 1.01x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.69% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 2.85% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 42.65% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 79.97% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[16384, 1, 1]` / `[256, 1, 1]`; registers/thread=24, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=4.00% and shared-memory=4.00%; digest effective bound=4.00%.
- CycleTrace WIF mean/P25/P50/P75/max=397.69/409.00/412.00/413.00/416 raw waves; P75/P25=1.01x.
- mcProfiler achieved/dispatched waves=65,484/66,112, workgroups=16,528, average wave life=9,865 cycles.
- DPC compute balance: min/avg/max=28,409,792/30,458,356/30,751,008 cycles, imbalance=7.69%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 127,359 | 54.12% | Vector/Matrix Tensor Extension compute category |
| STE | 66,788 | 28.38% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 954 | 0.41% | Block shared-memory hardware category |
| GLOBAL | 18,444 | 7.84% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 20,829 | 8.85% | Synchronization/arrival category |
| LDU | 953 | 0.41% | Load data unit category |

- STE subevents by `name`: STE=27,181, S_NOP=31,180, Branch=8,427.
- GLOBAL subevents by `name`: GVM Load=15,264, GVM Store=3,180.
- ARRIVE subevents by `name`: Synchronization=20,829.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 127,359 / 54.12% | 53,110,112 / 61.38% | +7.26 pp |
| STE | 66,788 / 28.38% | 11,185,340 / 12.93% | -15.45 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 954 / 0.41% | 1,703,940 / 1.97% | +1.56 pp |
| GLOBAL/GVM | 18,444 / 7.84% | 7,530,364 / 8.70% | +0.87 pp |
| LDU | 953 / 0.41% | 392,332 / 0.45% | +0.05 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=65.44% / 13.39% / 0.00%.
- Real IPC=105.71; instruction throughput=1.016; throughput efficiency=25.41%; compute-instruction busy duty=72.92%.
- Instructions per AP=831,939; average cycles/instruction=2.00; average all-stage latency/instruction=39.20 cycles.
- Compute instruction cycle split: total=343,575,812 cycles, MTE=343,575,812 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=33,889,167 cycles, top=vls_pipeline_stall 27,100,387 cycles (79.97%).
- AP active cycles=83,540,207; average AP busy cycles=811,070; AP busy duty=0.02092. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 1,586,668 cycles (4.68%)
- vls_pipeline_stall: 27,100,387 cycles (79.97%)
- vls_wdata_stall: 5,202,112 cycles (15.35%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=98.22% / 39.64% / 99.91%.
- DNOC read average latency=282.8 cycles; read/write requests=3,417,618/1,055,317; achieved bandwidth=786.13 GB/s.
- Global memory bytes read/write=436,851,296/135,072,576; global instructions read/write=6,231,552/1,298,812.
- VL1 instructions read/write=6,231,552/1,298,812; L2C instructions read/write=6,306,757/2,368,606.
- Constant read path: total=392,332, SL1=240,568, L2=204.
- DNOC latency histogram total=3,410,348 samples, >512-cycle samples=97,156 (2.85%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 1,662,853 samples (48.76%)
- 256-511: 1,650,339 samples (48.39%)
- 512-1k: 92,806 samples (2.72%)
- 1k-2k: 982 samples (0.03%)
- 2k-: 3,368 samples (0.10%)

Memory data flow:

- global: read=6,231,552, write=1,298,812
- shared: read=1,070,982, write=632,853
- constant: read=392,332, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=6,231,552, write=1,298,812
- const_sl1c: read=240,568, write=0
- vl1c_l2c: read=6,306,757, write=2,368,606
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=1,070,982/632,853.
- Avg cycles per load/store=2.00/2.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.65/30.03/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 5.02 TFLOP/s @ 786.13 GByte/s.
- Intensities: HBM=6.39, VL1=1.89, L2C=4.58 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=42.65%, VL1=17.69%, L2C=23.79%.
- Roofline raw context: all_ops=3,652,250,112, all_memacs=571,923,872, vl1_memacs=1,927,653,376, l2c_memacs=797,638,656, duration=818,462, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=4.00%, smem=4.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

### MEDIUM: Pipeline bubble signal

- Evidence: NOP share=13.25%
- Next: Correlate with barriers, load-use distance, and mcProfiler IPC.

## 5. Inference Chain

1. Measured: MTE share=54.12% / AP MTE duty=65.44%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=4.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=54.12%, MMA share=0.00%, AP MTE duty=65.44%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=7.84%, L2C hit=39.64%, DNOC >512-cycle share=2.85%, DPC imbalance=7.69%, WIF P75/P25=1.01x, and effective occupancy=4.00%.
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
