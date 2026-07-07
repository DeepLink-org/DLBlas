# `pre_split_mixes_kernel_opt(float const*, float const*, float const*, float*, float*, float*, int, int, int, float, float)` Trace Profiling Report

**Tag:** `baseline`
**Target GPU / arch:** MetaX C500
**Tools:** CycleTrace + mcTracer + mcProfiler
**Run directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run/profile-artifacts/pre_split_mixes_v0_baseline`
**Artifact directory:** `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run/profile-artifacts/pre_split_mixes_v0_baseline/artifacts`

## 0. Profiling setup

- Kernel: `pre_split_mixes_kernel_opt(float const*, float const*, float const*, float*, float*, float*, int, int, int, float, float)`
- Grid / block: `[2, 1, 1]` / `[512, 1, 1]`
- CycleTrace JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run/profile-artifacts/pre_split_mixes_v0_baseline/artifacts/c-trace_output_dpc_2.json`
- CycleTrace DPC selection: archived `c-trace_output_dpc_2.json, c-trace_output_dpc_3.json`; analysis uses primary `c-trace_output_dpc_2.json`.
- mcTracer JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run/profile-artifacts/pre_split_mixes_v0_baseline/artifacts/tracer_out.json`
- mcProfiler JSON: `/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run/profile-artifacts/pre_split_mixes_v0_baseline/artifacts/mcprofiler_report_dumped.json`
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
- Dominant signal: MTE share=59.56% / AP MTE duty=20.34%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=4.00%
- One-line read: this kernel is classified as `compute` because compute-side evidence dominates (MTE share=59.56%, AP MTE duty=20.34%, AP MMA duty=0.00%).

## 2. Evidence

| Metric | Value | Source | Interpretation |
|---|---:|---|---|
| Kernel span | 26,163,126 cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |
| CycleTrace MTE share | 59.56% | CycleTrace instruction events | Vector compute path dominates instruction mix |
| CycleTrace MMA share | 0.00% | CycleTrace instruction events | MMA path is not expected for this kernel type |
| AP MTE duty | 20.34% | mcProfiler SOL | Hardware vector pipe duty |
| AP MMA duty | 0.00% | mcProfiler SOL | Tensor pipe utilization |
| Shared memory efficiency | 0.00% | mcProfiler WSM | Bank conflict signal when below 80-85% |
| Avg conflict cycles/inst | 0 | mcProfiler WSM | Conflict penalty per smem instruction |
| Effective occupancy bound | 4.00% | mcTracer | Lower of mtreg and shared-memory occupancy fields |
| L2C hit rate | 78.84% | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |
| WIF P75/P25 | 3.00x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |
| DPC compute imbalance | 8.04% | mcProfiler DPC cycles | Aggregate compute-balance signal |
| DNOC >512-cycle share | 1.44% | mcProfiler DNOC histogram | DRAM latency-tail check |
| Roofline HBM usage | 45.55% | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |
| Top ISU stall | vls_pipeline_stall / 76.14% | mcProfiler ISU stall layout | Dominant issue-side stall bucket |

## 3. Per-dimension analysis

### 3.1 Occupancy & Launch
- Grid/block is `[2, 1, 1]` / `[512, 1, 1]`; registers/thread=22, static shared=0 bytes.
- mcTracer occupancy fields are mtreg=4.00% and shared-memory=0.00%; digest effective bound=4.00%.
- CycleTrace WIF mean/P25/P50/P75/max=4.01/2.00/4.00/6.00/12 raw waves; P75/P25=3.00x.
- mcProfiler achieved/dispatched waves=8,292/8,376, workgroups=1,084, average wave life=2,597 cycles.
- DPC compute balance: min/avg/max=1,937,888/2,072,015/2,104,496 cycles, imbalance=8.04%.

### 3.2 Instruction Distribution
- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.

| CycleTrace cat | Count | Share | Hardware meaning |
|---|---:|---:|---|
| MTE | 4,878 | 59.56% | Vector/Matrix Tensor Extension compute category |
| STE | 2,592 | 31.65% | Scalar/control hardware category; includes STE-name, S_NOP, and Branch events |
| MMA | 0 | 0.00% | Matrix-multiply hardware category |
| BSM | 0 | 0.00% | Block shared-memory hardware category |
| GLOBAL | 216 | 2.64% | Global vector-memory hardware category; GVM Load/Store subevents |
| ARRIVE | 288 | 3.52% | Synchronization/arrival category |
| LDU | 216 | 2.64% | Load data unit category |

- STE subevents by `name`: STE=1,764, S_NOP=594, Branch=234.
- GLOBAL subevents by `name`: GVM Load=108, GVM Store=108.
- ARRIVE subevents by `name`: Synchronization=288.

Hardware instruction cross-check:

| Class | CycleTrace count/share | mcProfiler count/share | Delta |
|---|---:|---:|---:|
| MTE | 4,878 / 59.56% | 2,217,438 / 64.28% | +4.72 pp |
| STE | 2,592 / 31.65% | 794,793 / 23.04% | -8.61 pp |
| MMA | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| BSM | 0 / 0.00% | 0 / 0.00% | +0.00 pp |
| GLOBAL/GVM | 216 / 2.64% | 96,286 / 2.79% | +0.15 pp |
| LDU | 216 / 2.64% | 97,480 / 2.83% | +0.19 pp |

### 3.3 SOL / Hardware Duty
- AP MTE/STE/MMA duty=20.34% / 6.22% / 0.00%.
- Real IPC=25.07; instruction throughput=0.241; throughput efficiency=6.03%; compute-instruction busy duty=32.41%.
- Instructions per AP=33,172; average cycles/instruction=0.00; average all-stage latency/instruction=0.00 cycles.
- Compute instruction cycle split: total=41,266,296 cycles, MTE=41,266,296 (100.00%), MMA=0 (0.00%).
- ISU stall summary: total=3,238,433 cycles, top=vls_pipeline_stall 2,465,869 cycles (76.14%).
- AP active cycles=12,787,888; average AP busy cycles=124,154; AP busy duty=0.003971. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.

ISU stall cycle layout:

- wsm_stall: 0 cycles (0.00%)
- vls_pipeline_stall: 2,465,869 cycles (76.14%)
- vls_wdata_stall: 772,564 cycles (23.86%)
- valu_stall: 0 cycles (0.00%)

### 3.4 Memory Hierarchy
- VL1/L2C/SL1 hit rate=25.15% / 78.84% / 94.66%.
- DNOC read average latency=181.7 cycles; read/write requests=443,001/394,673; achieved bandwidth=839.62 GB/s.
- Global memory bytes read/write=52,919,424/49,788,864; global instructions read/write=48,048/48,334.
- VL1 instructions read/write=48,000/48,286; L2C instructions read/write=2,433,159/2,339,246.
- Constant read path: total=97,480, SL1=59,260, L2=3,117.
- DNOC latency histogram total=444,703 samples, >512-cycle samples=6,419 (1.44%).
- VL1 partition stalls: min/avg/max=20,201/44,722/64,750 cycles, spread=99.61%.

DNOC latency buckets:

- 0-255: 422,300 samples (94.96%)
- 256-511: 15,984 samples (3.59%)
- 512-1k: 5,575 samples (1.25%)
- 1k-2k: 844 samples (0.19%)
- 2k-: 0 samples (0.00%)

Memory data flow:

- global: read=48,048, write=48,334
- shared: read=0, write=0
- constant: read=97,480, write=0
- private: read=0, write=0
- generic: read=0, write=0
- generic_vl1c: read=48,000, write=48,286
- const_sl1c: read=59,260, write=0
- vl1c_l2c: read=2,433,159, write=2,339,246
- l2c_global: read=0, write=0

### 3.5 Bank Conflict & Private Memory
- Shared memory load/store instructions=0/0.
- Avg cycles per load/store=0.00/0.00; conflict cycles/inst=0.
- Avg latency per load/store/atomic instruction=0.00/0.00/0.00 cycles; avg cycles per atomic instruction=0.00.
- Private memory from mcTracer: per_thread=0, total=0.
- Private memory from mcProfiler memory flow: read=0, write=0.
- Atomic pressure: 0 total atomic scalar counts across 7 atomic-related mcProfiler fields.

### 3.6 Roofline
- Achieved point: 1.36 TFLOP/s @ 839.62 GByte/s.
- Intensities: HBM=1.62, VL1=6.75, L2C=0.56 FLOP/Byte.
- Peak context from mcProfiler RoofLine: TRANS=1.87, FMA=14.98, MMA-FP16=239.62, INT8=479.23 TFLOP/s.
- Bandwidth usage from mcProfiler RoofLine context: HBM=45.55%, VL1=1.35%, L2C=52.32%.
- Roofline raw context: all_ops=166,404,480, all_memacs=102,708,288, vl1_memacs=24,648,740, l2c_memacs=294,912,000, duration=137,618, core_clk=1.125 GHz, mc_clk=0.900 GHz.
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

- Evidence: mtreg=4.00%, smem=0.00%
- Next: For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter. streg limit still needs compiler/assembly evidence.

## 5. Inference Chain

1. Measured: MTE share=59.56% / AP MTE duty=20.34%; MMA share=0.00% / AP MMA duty=0.00%; SMEM efficiency=0.00%, conflict cycles/inst=0; effective occupancy=4.00%.
2. Likely mechanism: compute-side evidence is the current primary signal: MTE share=59.56%, MMA share=0.00%, AP MTE duty=20.34%, AP MMA duty=0.00%, and MMA compute-cycle share=0.00%.
3. Why other hypotheses are weaker: counter-evidence for alternate bounds is GVM share=2.64%, L2C hit=78.84%, DNOC >512-cycle share=1.44%, DPC imbalance=8.04%, WIF P75/P25=3.00x, and effective occupancy=4.00%.
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
