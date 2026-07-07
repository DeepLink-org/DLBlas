# `engram_gate_fwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, float*, float*, float*, float*, int, int, int, float, float, float)` Trace Profiling Report

**Tag:** `opt8x`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v1_opt8x`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v1_opt8x/artifacts`

## 0. Profiling setup

- Kernel: `engram_gate_fwd_kernel_opt(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*, float*, float*, float*, float*, int, int, int, float, float, float)`
- Grid / block: `[16384, 1, 1]` / `[64, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v1_opt8x/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v1_opt8x/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run/profile-artifacts/engram_gate_fwd_v1_opt8x/artifacts/mcprofiler_report_dumped.json`
- Bound mode: `coarse`

Runnable pipeline:

```bash
python3 scripts/trace_report_env.py --config "$ACTIVE_CONFIG" run -- \
  python3 .trace-report/scripts/trace_profile_pipeline.py run \
  --source <profile-artifacts-dir> \
  --run-dir <profile-artifacts-dir> \
  --tag opt8x \
  --cycle-dpc-id 2,3 \
  --bound-mode coarse
```

## 1. Headline

- Bottleneck class: `memory`
- Primary: `memory`
- Confidence: Medium
- Dominant signal: MTE share=66.18% / AP MTE duty=73.66%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%
- One-line read: this kernel is classified as `memory` because GVM/VLS/L2C/cache thresholds fired (GVM share=12.94%, L2C hit=38.98%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 420,988,572 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 66.18% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 73.66% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 100.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 1.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 38.98% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 1.02x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 7.68% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 14.92% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 68.53% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 84.46% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[16384, 1, 1]` / `[64, 1, 1]`; registers/thread=60, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=11.00% and shared-memory=1.00%; digest effective bound=1.00%.
- CycleTrace WIF mean/P25/P50/P75/max=355.71/403.00/407.00/410.00/416 raw waves; P75/P25=1.02x.
- mcProfiler achieved/dispatched waves=16,800/16,960, workgroups=16,528, average wave life=22,137 cycles.
- DPC compute balance: min/avg/max=18,457,872/19,788,446/19,978,528 cycles, imbalance=7.68%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 90,168 | 66.18% | Vector/Matrix Tensor Extension compute category |
| STE | 14,875 | 10.92% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 2,340 | 1.72% | Block shared-memory hardware category |
| GLOBAL | 17,628 | 12.94% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 10,998 | 8.07% | Synchronization/arrival category |
| LDU | 233 | 0.17% | Load data unit category |

- STE subevents by `name`: STE=4,764, S_NOP=8,512, Branch=1,599.
- GLOBAL subevents by `name`: GVM Load=14,976, GVM Store=2,652.
- ARRIVE subevents by `name`: Synchronization=10,998.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 90,168 / 66.18% | 37,612,372 / 70.56% | +4.38 pp |
| STE | 14,875 / 10.92% | 2,017,208 / 3.78% | -7.13 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 2,340 / 1.72% | 973,680 / 1.83% | +0.11 pp |
| GLOBAL/GVM | 17,628 / 12.94% | 7,335,628 / 13.76% | +0.82 pp |
| LDU | 233 / 0.17% | 100,208 / 0.19% | +0.02 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=73.66% / 3.91% / 0.00%.
- Real IPC=103.93; instruction throughput=0.9993; throughput efficiency=24.98%; compute-instruction busy duty=76.71%.
- Instructions per AP=512,542; average cycles/instruction=2.00; average all-stage latency/instruction=39.83 cycles.
- Compute instruction cycle split: total=183,450,848 cycles, MTE=183,450,848 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=34,120,304 cycles, top=vls_pipeline_stall 28,816,873 cycles (84.46%).
- AP active cycles=51,592,451; average AP busy cycles=500,898; AP busy duty=0.01311. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 880,327 cycles (2.58%)
- vls_pipeline_stall: 28,816,873 cycles (84.46%)
- vls_wdata_stall: 4,423,104 cycles (12.96%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=98.21% / 38.98% / 99.66%.
- DNOC read average latency=358 cycles; read/write requests=3,445,230/1,055,335; achieved bandwidth=1263.10 GB/s.
- Global memory bytes read/write=440,762,112/135,073,280; global instructions read/write=6,231,552/1,104,076.
- VL1 instructions read/write=6,231,552/1,104,076; L2C instructions read/write=6,311,298/2,171,998.
- Constant read path: total=100,208, SL1=61,498, L2=198.
- DNOC latency histogram total=3,439,763 samples, >512-cycle samples=513,272 (14.92%).
- VL1 partition stalls: min/avg/max=0/13.25/53 cycles, spread=400.00%.

DNOC latency buckets:

- 0-255: 1,058,583 samples (30.77%)
- 256-511: 1,867,908 samples (54.30%)
- 512-1k: 505,537 samples (14.70%)
- 1k-2k: 7,735 samples (0.22%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=6,231,552, write=1,104,076
- shared: read=632,853, write=340,788
- constant: read=100,208, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=6,231,552, write=1,104,076
- const_sl1c: read=61,498, write=0
- vl1c_l2c: read=6,311,298, write=2,171,998
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=632,853/340,788.
- Avg cycles per load/store=2.00/2.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=34.74/30.28/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 5.80 TFLOP/s @ 1263.10 GByte/s.
- Intensities: HBM=4.59, VL1=1.41, L2C=3.32 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=68.53%, VL1=27.51%, L2C=37.97%.
- Roofline raw context: all_ops=2,644,795,072, all_memacs=575,835,392, vl1_memacs=1,877,920,768, l2c_memacs=797,638,656, duration=512,878, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=11.00%, smem=1.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## 5. Inference Chain

1. Measured: MTE share=66.18% / AP MTE duty=73.66%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=100.00%, conflict cycles/inst=0; effective occupancy=1.00%.
2. Likely mechanism: memory/cache evidence is the current primary signal: GVM share=12.94%, VLS duty=0.00%, L2C duty=0.05%, L2C hit=38.98%, and DNOC >512-cycle share=14.92%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=12.94%, L2C hit=38.98%, DNOC >512-cycle share=14.92%, DPC imbalance=7.68%, WIF P75/P25=1.02x, and effective occupancy=1.00%.
4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.

## 6. Next Concrete Edit

- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing.
- Change: reduce global read/write traffic for this non-tensor kernel, for example by removing redundant loads/stores, improving contiguous/vectorized access, or fusing adjacent elementwise work when that path exists.
- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare `cycle.gvm_pct`, `profiler.vls_duty_pct`, `profiler.l2c_hit_rate_pct`, DNOC latency buckets, global memory bytes, and `REPORT_<tag>.md`.
- Expected metric movement: global bytes, GVM/VLS pressure, and DNOC requests should decrease without introducing shared-memory traffic or occupancy regressions.

## 7. Artifacts

- Full metrics: `analysis/metrics_all_opt8x.json`
- Key metrics: `analysis/metrics_key_opt8x.json`
- Digest: `analysis/digest_opt8x.md`
- This report: `REPORT_opt8x.md`

## 8. Caveats

- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.
- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.
- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.
- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.
