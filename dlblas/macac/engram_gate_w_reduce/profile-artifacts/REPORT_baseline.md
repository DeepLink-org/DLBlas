# `_Z31engram_gate_w_reduce_kernel_oriPKfPKDF16_S2_S0_S0_PfS3_iii` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run/profile-artifacts/engram_gate_w_reduce_v0_baseline`
**Artifact directory:** `/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run/profile-artifacts/engram_gate_w_reduce_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `_Z31engram_gate_w_reduce_kernel_oriPKfPKDF16_S2_S0_S0_PfS3_iii`
- Grid / block: `[64, 1, 1]` / `[256, 1, 1]`
- CycleTrace JSON: `/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run/profile-artifacts/engram_gate_w_reduce_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run/profile-artifacts/engram_gate_w_reduce_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run/profile-artifacts/engram_gate_w_reduce_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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

- Bottleneck class: `memory`
- Primary: `memory`
- Confidence: Medium
- Dominant signal: MTE share=52.12% / AP MTE duty=11.01%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=5.00%
- One-line read: this kernel is classified as `memory` because GVM/VLS/L2C/cache thresholds fired (GVM share=16.10%, L2C hit=2.48%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 115,187,543 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 52.12% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 11.01% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 5.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 2.48% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 3.00x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.76% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 1.18% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 52.78% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 95.52% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[64, 1, 1]` / `[256, 1, 1]`; registers/thread=26, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=5.00% and shared-memory=0.00%; digest effective bound=5.00%.
- CycleTrace WIF mean/P25/P50/P75/max=15.99/8.00/16.00/24.00/32 raw waves; P75/P25=3.00x.
- mcProfiler achieved/dispatched waves=510,056/515,008, workgroups=128,752, average wave life=3,294 cycles.
- DPC compute balance: min/avg/max=87,676,032/94,066,612/94,979,552 cycles, imbalance=7.76%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 456,084 | 52.12% | Vector/Matrix Tensor Extension compute category |
| STE | 159,444 | 18.22% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 140,904 | 16.10% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 111,240 | 12.71% | Synchronization/arrival category |
| LDU | 7,416 | 0.85% | Load data unit category |

- STE subevents by `name`: STE=84,048, S_NOP=49,440, Branch=25,956.
- GLOBAL subevents by `name`: GVM Load=138,432, GVM Store=2,472.
- ARRIVE subevents by `name`: Synchronization=111,240.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 456,084 / 52.12% | 188,120,024 / 55.16% | +3.04 pp |
| STE | 159,444 / 18.22% | 34,670,168 / 10.17% | -8.05 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 140,904 / 16.10% | 58,096,664 / 17.04% | +0.93 pp |
| LDU | 7,416 / 0.85% | 3,059,896 / 0.90% | +0.05 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=11.01% / 2.03% / 0.00%.
- Real IPC=19.86; instruction throughput=0.1909; throughput efficiency=4.77%; compute-instruction busy duty=11.01%.
- Instructions per AP=3,279,129; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=752,701,856 cycles, MTE=752,701,856 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=91,228,964 cycles, top=vls_pipeline_stall 87,144,996 cycles (95.52%).
- AP active cycles=1,709,360,064; average AP busy cycles=16,595,729; AP busy duty=0.4791. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 87,144,996 cycles (95.52%)
- vls_wdata_stall: 4,083,968 cycles (4.48%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=96.85% / 2.48% / 95.93%.
- DNOC read average latency=241.7 cycles; read/write requests=114,007,502/2,083,735; achieved bandwidth=972.84 GB/s.
- Global memory bytes read/write=14,586,605,856/263,828,576; global instructions read/write=57,076,992/1,019,672.
- VL1 instructions read/write=57,076,992/1,019,672; L2C instructions read/write=116,943,652/4,153,750.
- Constant read path: total=3,059,896, SL1=1,871,580, L2=88,844.
- DNOC latency histogram total=113,904,652 samples, >512-cycle samples=1,349,028 (1.18%).
- VL1 partition stalls: min/avg/max=0/0/0 cycles, spread=0.00%.

DNOC latency buckets:

- 0-255: 81,804,403 samples (71.82%)
- 256-511: 30,751,221 samples (27.00%)
- 512-1k: 1,329,260 samples (1.17%)
- 1k-2k: 16,444 samples (0.01%)
- 2k-: 3,324 samples (0.00%)

Memory data flow:

- global: read=57,076,992, write=1,019,672
- shared: read=0, write=0
- constant: read=3,059,896, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=57,076,992, write=1,019,672
- const_sl1c: read=1,871,580, write=0
- vl1c_l2c: read=116,943,652, write=4,153,750
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 0.79 TFLOP/s @ 972.84 GByte/s.
- Intensities: HBM=0.82, VL1=0.81, L2C=0.84 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=52.78%, VL1=6.51%, L2C=20.59%.
- Roofline raw context: all_ops=12,104,743,424, all_memacs=14,850,434,432, vl1_memacs=14,872,745,984, l2c_memacs=14,481,248,256, duration=17,173,234, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

1. Measured: MTE share=52.12% / AP MTE duty=11.01%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=5.00%.
2. Likely mechanism: memory/cache evidence is the current primary signal: GVM share=16.10%, VLS duty=0.00%, L2C duty=2.23%, L2C hit=2.48%, and DNOC >512-cycle share=1.18%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=16.10%, L2C hit=2.48%, DNOC >512-cycle share=1.18%, DPC imbalance=7.76%, WIF P75/P25=3.00x, and effective occupancy=5.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: reduce global read/write traffic for this non-tensor kernel, for example by removing redundant loads/stores, improving contiguous/vectorized access, or fusing adjacent elementwise work when that path exists.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare `cycle.gvm_pct`, `profiler.vls_duty_pct`, `profiler.l2c_hit_rate_pct`, DNOC latency buckets, global memory bytes, and `REPORT_<tag>.md`.
- Expected metric movement: global bytes, GVM/VLS pressure, and DNOC requests should decrease without introducing shared-memory traffic or occupancy regressions.

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
