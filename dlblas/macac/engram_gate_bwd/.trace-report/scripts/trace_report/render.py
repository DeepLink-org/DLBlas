"""Markdown rendering for trace-report metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import KEY_METRICS, NA, cycle_trace_filename
from .utils import (
    as_number,
    expects_tensor_path,
    fmt,
    fmt_float,
    fmt_pct,
    get_path,
    min_avg_max,
    numeric_values,
    parse_number,
    pct,
    ratio,
)

def dpc_balance_summary(profiler: dict[str, Any]) -> str:
    values = numeric_values(profiler.get("dpc_compute_cycles", {}))
    if not values:
        return NA
    lo, avg, hi = min_avg_max(values)
    return (
        f"min/avg/max={fmt(lo)}/{fmt(avg)}/{fmt(hi)} cycles, "
        f"imbalance={profiler.get('dpc_compute_imbalance_pct', 0):.2f}%"
    )

def high_dnoc_share(histogram: dict[str, Any]) -> tuple[float, float, float]:
    total = sum(parse_number(v) for v in histogram.values())
    high = sum(
        parse_number(v)
        for k, v in histogram.items()
        if "512" in str(k) or "1k" in str(k) or "2k" in str(k)
    )
    return total, high, pct(high, total)

def vl1_partition_summary(profiler: dict[str, Any]) -> str:
    summary = profiler.get("vl1_partition_stall_summary", {})
    if summary:
        return (
            f"min/avg/max={fmt(summary.get('min'))}/{fmt(summary.get('avg'))}/{fmt(summary.get('max'))} cycles, "
            f"spread={fmt_pct(summary.get('spread_pct'))}"
        )
    values = numeric_values(profiler.get("vl1_partition_stalls", {}))
    if not values:
        return NA
    lo, avg, hi = min_avg_max(values)
    spread = pct(hi - lo, avg)
    return f"min/avg/max={fmt(lo)}/{fmt(avg)}/{fmt(hi)} cycles, spread={spread:.2f}%"

def dnoc_histogram_rows(profiler: dict[str, Any]) -> list[str]:
    histogram = profiler.get("dnoc_latency_histogram", {})
    shares = profiler.get("dnoc_latency_histogram_pct", {})
    if not histogram:
        return ["- DNOC latency histogram is unavailable."]
    return [
        f"- {bucket}: {fmt(value)} samples ({shares.get(bucket, 0):.2f}%)"
        for bucket, value in histogram.items()
    ]

def compute_cycles_summary(profiler: dict[str, Any]) -> str:
    summary = profiler.get("compute_instruction_cycles_summary", {})
    if not summary:
        return NA
    return (
        f"total={fmt(summary.get('total'))} cycles, "
        f"MTE={fmt(summary.get('mte_cycles'))} ({fmt_pct(summary.get('mte_cycles_pct'))}), "
        f"MMA={fmt(summary.get('mma_cycles'))} ({fmt_pct(summary.get('mma_cycles_pct'))})"
    )

def isu_stall_summary(profiler: dict[str, Any]) -> str:
    summary = profiler.get("isu_stall_summary", {})
    if not summary:
        return NA
    return (
        f"total={fmt(summary.get('total'))} cycles, "
        f"top={summary.get('top', '')} {fmt(summary.get('top_cycles'))} cycles "
        f"({fmt_pct(summary.get('top_pct'))})"
    )

def stall_share_rows(profiler: dict[str, Any]) -> list[str]:
    stalls = profiler.get("isu_stall_cycles", {})
    shares = profiler.get("isu_stall_summary", {}).get("shares_pct", {})
    if not stalls:
        return ["- ISU stall cycle layout is unavailable."]
    return [
        f"- {name}: {fmt(value)} cycles ({shares.get(name, 0):.2f}%)"
        for name, value in stalls.items()
    ]

def roofline_peak_summary(roof: dict[str, Any]) -> str:
    return (
        f"TRANS={fmt_float(roof.get('peak_trans_tflops'))}, "
        f"FMA={fmt_float(roof.get('peak_fma_tflops'))}, "
        f"MMA-FP16={fmt_float(roof.get('peak_mma_fp16_tflops'))}, "
        f"INT8={fmt_float(roof.get('peak_int8_tflops'))} TFLOP/s"
    )

def roofline_usage_summary(roof: dict[str, Any]) -> str:
    return (
        f"HBM={fmt_pct(roof.get('hbm_usage_pct'))}, "
        f"VL1={fmt_pct(roof.get('vl1_usage_pct'))}, "
        f"L2C={fmt_pct(roof.get('l2c_usage_pct'))}"
    )

def memory_flow_summary(flow: dict[str, Any]) -> list[str]:
    if not flow:
        return ["- Memory data flow chart is unavailable."]
    aliases = (
        ("global", ("global_kernel_rd", "global_kernel_wr")),
        ("shared", ("shared_kernel_rd", "shared_kernel_wr")),
        ("constant", ("const_kernel_rd", "const_kernel_wr")),
        ("private", ("private_kernel_rd", "private_kernel_wr")),
        ("generic", ("generic_kernel_rd", "generic_kernel_wr")),
        ("generic_vl1c", ("Ggeneric_vl1c_rd", "Ggeneric_vl1c_wr")),
        ("const_sl1c", ("const_sl1c_rd", "const_sl1c_wr")),
        ("vl1c_l2c", ("vl1c_l2c_rd", "vl1c_l2c_wr")),
        ("l2c_global", ("l2c_Gmemory_rd", "l2c_Gmemory_wr")),
    )
    lines = []
    for label, (rd_key, wr_key) in aliases:
        rd_present = rd_key in flow
        wr_present = wr_key in flow
        rd = flow.get(rd_key, 0)
        wr = flow.get(wr_key, 0)
        if rd_present or wr_present or parse_number(rd) or parse_number(wr):
            lines.append(f"- {label}: read={fmt(parse_number(rd))}, write={fmt(parse_number(wr))}")
    return lines or ["- Memory data flow chart has no recognized read/write entries."]

def render_run_dir(artifact_dir: str) -> str:
    if not artifact_dir:
        return ""
    path = Path(artifact_dir)
    if path.name == "artifacts" and path.parent != path:
        return str(path.parent)
    if path.parent != path:
        return str(path.parent)
    return str(path)

def dpc_id_for_rerun(metrics: dict[str, Any]) -> str:
    explicit = metrics.get("cycle_dpc_id")
    if explicit:
        return str(explicit)
    primary = metrics.get("cycle_dpc_primary")
    if primary is not None:
        return str(primary)
    filename = str(metrics.get("cycle_trace_file", ""))
    prefix = "c-trace_output_dpc_"
    suffix = ".json"
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix):-len(suffix)]
    return "0"

def metric_coverage_rows(coverage: dict[str, Any]) -> list[str]:
    rows = [
        "| Category | Count | Meaning |",
        "|---|---:|---|",
    ]
    summary = coverage.get("summary", {})
    rows.extend([
        f"| Reported metric groups | {summary.get('reported_metric_groups', 0)} | Metric groups directly shown in `REPORT_<tag>.md` |",
        f"| Parsed but not promoted groups | {summary.get('parsed_not_promoted_groups', 0)} | Available in `metrics_all_<tag>.json`, omitted from the main prose unless diagnostic |",
        f"| Unavailable dimensions | {summary.get('unavailable_dimensions', 0)} | Analyses that current artifacts cannot support |",
    ])
    return rows

def hardware_instruction_rows(cycle: dict[str, Any], profiler: dict[str, Any]) -> list[str]:
    mapping = (
        ("MTE", "MTE", "mte", "mte_pct"),
        ("STE", "STE", "ste", "ste_pct"),
        ("MMA", "MMA", "mma", "mma_pct"),
        ("BSM", "BSM", "bsm", "bsm_pct"),
        ("GLOBAL/GVM", "GVM", "gvm_total", "gvm_pct"),
        ("LDU", "LDU", "ldu", None),
    )
    hw_counts = profiler.get("hardware_instruction_counts", {})
    hw_total = as_number(profiler.get("hardware_instruction_total", 0))
    rows = [
        "| Class | CycleTrace count/share | mcProfiler count/share | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, hw_label, count_key, pct_key in mapping:
        cycle_count = cycle.get(count_key, 0)
        cycle_pct = cycle.get(pct_key, pct(cycle_count, cycle.get("total_instructions", 0)))
        hw_count = parse_number(hw_counts.get(hw_label, 0))
        hw_pct = pct(hw_count, hw_total)
        rows.append(
            f"| {label} | {fmt(cycle_count)} / {cycle_pct:.2f}% | "
            f"{fmt(hw_count)} / {hw_pct:.2f}% | {hw_pct - cycle_pct:+.2f} pp |"
        )
    return rows

def cycle_category_rows(cycle: dict[str, Any]) -> list[str]:
    category_counts = cycle.get("category_counts", {})
    category_pcts = cycle.get("category_pcts", {})
    category_meaning = {
        "MTE": "Vector/Matrix Tensor Extension compute category",
        "STE": "Scalar/control hardware category; includes STE-name, S_NOP, and Branch events",
        "MMA": "Matrix-multiply hardware category",
        "BSM": "Block shared-memory hardware category",
        "GLOBAL": "Global vector-memory hardware category; GVM Load/Store subevents",
        "ARRIVE": "Synchronization/arrival category",
        "LDU": "Load data unit category",
    }
    rows = [
        "| CycleTrace cat | Count | Share | Hardware meaning |",
        "|---|---:|---:|---|",
    ]
    for cat in ("MTE", "STE", "MMA", "BSM", "GLOBAL", "ARRIVE", "LDU"):
        rows.append(
            f"| {cat} | {fmt(category_counts.get(cat, 0))} | "
            f"{category_pcts.get(cat, 0):.2f}% | {category_meaning[cat]} |"
        )
    return rows

def render_digest(metrics: dict[str, Any]) -> str:
    tag = metrics["tag"]
    kernel = metrics["kernel"]
    launch = metrics["launch"]
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    bound = metrics["bound"]

    lines = [
        f"# C500 Trace Profile Digest: {tag}",
        "",
        "## Kernel",
        "",
        f"- Name: `{kernel.get('name', '')}`",
        f"- Grid: `{kernel.get('grid', [])}`",
        f"- Block: `{kernel.get('block', [])}`",
        f"- Span: `{fmt(kernel.get('span_cycles', 0))}` cycles",
        "",
        "## Bound Classification",
        "",
        f"- Mode: `{bound.get('mode', '')}`",
        f"- Type: `{bound.get('type', '')}`",
        f"- Primary: `{bound.get('primary', '')}`",
        "",
        "Evidence:",
        "",
        f"- MMA share / AP MMA duty: {cycle.get('mma_pct', 0):.2f}% / {fmt_pct(profiler.get('ap_mma_duty_pct'))}",
        f"- MTE share / AP MTE duty: {cycle.get('mte_pct', 0):.2f}% / {fmt_pct(profiler.get('ap_mte_duty_pct'))}",
        f"- GVM share / VLS duty / L2C duty: {cycle.get('gvm_pct', 0):.2f}% / {fmt_pct(profiler.get('vls_duty_pct'))} / {fmt_pct(profiler.get('l2c_duty_pct'))}",
        f"- VL1/L2C hit rate: {fmt_pct(profiler.get('vl1_hit_rate_pct'))} / {fmt_pct(profiler.get('l2c_hit_rate_pct'))}",
        f"- Shared-memory efficiency / conflict cycles: {fmt_pct(profiler.get('shared_memory_efficiency_pct'))} / {fmt(profiler.get('avg_conflict_cycles_per_inst'))}",
        f"- Effective occupancy bound: {launch.get('effective_occupancy_pct', 0):.2f}%",
        f"- WIF P25/P50/P75/max: {cycle.get('wif_p25', 0):.2f} / {cycle.get('wif_p50', 0):.2f} / {cycle.get('wif_p75', 0):.2f} / {cycle.get('wif_max', 0)}",
        f"- DPC balance: {dpc_balance_summary(profiler)}",
        "",
        "Rationale:",
        "",
        *[f"- {line}" for line in bound.get("rationale", [])],
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Unit | Better |",
        "|---|---:|---|---|",
    ]
    for path, label, unit, better in KEY_METRICS:
        lines.append(f"| {label} | {fmt(get_path(metrics, path))} | {unit} | {better} |")

    lines.extend([
        "",
        "## Instruction Distribution",
        "",
        "Primary classification uses CycleTrace `cat`, which identifies the C500 hardware category.",
        "",
        *cycle_category_rows(cycle),
        "",
        f"- STE subevents by `name`: STE={cycle.get('raw_name_counts', {}).get('STE', 0):,}, S_NOP={cycle.get('s_nop', 0):,}, Branch={cycle.get('branch', 0):,}.",
        f"- GLOBAL subevents by `name`: GVM Load={cycle.get('gvm_load', 0):,}, GVM Store={cycle.get('gvm_store', 0):,}.",
        f"- ARRIVE subevents by `name`: Synchronization={cycle.get('raw_name_counts', {}).get('Synchronization', 0):,}.",
        "",
        "## mcProfiler Hardware Evidence",
        "",
        f"- AP MTE/STE/MMA duty: {fmt_pct(profiler.get('ap_mte_duty_pct'))} / {fmt_pct(profiler.get('ap_ste_duty_pct'))} / {fmt_pct(profiler.get('ap_mma_duty_pct'))}",
        f"- Real IPC: {fmt_float(profiler.get('real_ipc'))}",
        f"- VL1/L2C hit rate: {fmt_pct(profiler.get('vl1_hit_rate_pct'))} / {fmt_pct(profiler.get('l2c_hit_rate_pct'))}",
        f"- Shared memory efficiency: {fmt_pct(profiler.get('shared_memory_efficiency_pct'))}",
        f"- Avg conflict cycles/inst: {fmt(profiler.get('avg_conflict_cycles_per_inst'))}",
        f"- DPC compute balance: {dpc_balance_summary(profiler)}",
        f"- Achieved/dispatched waves: {fmt(profiler.get('achieved_waves'))} / {fmt(profiler.get('dispatched_waves'))}; workgroups={fmt(profiler.get('workgroups'))}; avg wave life={fmt(profiler.get('average_wave_life_cycles'))} cycles",
        f"- Empirical roofline: {fmt_float(profiler.get('achieved_flops_tflops'))} TFLOPS, {fmt_float(profiler.get('achieved_bandwidth_gbs'))} GB/s",
        f"- Roofline intensity: DRAM={fmt_float(profiler.get('achieved_intensity_flop_per_byte'))}, VL1={fmt_float(profiler.get('roofline', {}).get('case_VL1_I', NA))}, L2C={fmt_float(profiler.get('roofline', {}).get('case_L2C_I', NA))} FLOP/Byte",
        f"- Roofline usage: {roofline_usage_summary(profiler.get('roofline', {}))}",
        "",
        "## Diagnosis",
        "",
    ])
    for item in metrics["diagnosis"]:
        lines.extend([
            f"### {item['severity'].upper()}: {item['title']}",
            "",
            f"- Evidence: {item['evidence']}",
            f"- Next: {item['next']}",
            "",
        ])

    lines.extend([
        "## Scope Notes",
        "",
        "- CycleTrace `dur=4` is used only as an issue-slot marker, not real instruction latency.",
        "- GVM 600-cycle peak is a pressure proxy, not exact buffer occupancy.",
        "- WIF is reported as raw trace distribution; converting it to waves/AP requires confirming counter scope.",
        "- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.",
        "",
    ])
    return "\n".join(lines)

def confidence(metrics: dict[str, Any]) -> str:
    bound = metrics.get("bound", {})
    evidence = bound.get("evidence", {})
    support = 0
    has_profiler = bool(metrics.get("profiler", {}).get("available", False))
    if evidence.get("cycle_mte_pct", 0) > 30 or (
        has_profiler and as_number(evidence.get("ap_mte_duty_pct", 0)) > 15
    ):
        support += 1
    if (
        has_profiler
        and as_number(evidence.get("shared_memory_efficiency_pct", 0))
        and as_number(evidence.get("shared_memory_efficiency_pct", 0)) < 80
    ):
        support += 1
    if has_profiler and as_number(evidence.get("avg_conflict_cycles_per_inst", 0)) > 0.5:
        support += 1
    if 0 < evidence.get("effective_occupancy_pct", 0) < 25:
        support += 1
    if (
        has_profiler
        and as_number(evidence.get("l2c_hit_rate_pct", 0)) > 90
        and evidence.get("cycle_gvm_pct", 0) < 15
    ):
        support += 1
    if support >= 3:
        return "High"
    if support >= 2:
        return "Medium"
    return "Low"

def primary_signal(metrics: dict[str, Any]) -> str:
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    launch = metrics["launch"]
    signals = [
        f"MTE share={cycle.get('mte_pct', 0):.2f}% / AP MTE duty={fmt_pct(profiler.get('ap_mte_duty_pct'))}",
        f"MMA share={cycle.get('mma_pct', 0):.2f}% / AP MMA duty={fmt_pct(profiler.get('ap_mma_duty_pct'))}",
        f"SMEM efficiency={fmt_pct(profiler.get('shared_memory_efficiency_pct'))}, conflict cycles/inst={fmt(profiler.get('avg_conflict_cycles_per_inst'))}",
        f"effective occupancy={launch.get('effective_occupancy_pct', 0):.2f}%",
    ]
    return "; ".join(signals)

def one_line_read(metrics: dict[str, Any]) -> str:
    bound = metrics["bound"]
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    primary = str(bound.get("primary", ""))
    if not profiler.get("available", False):
        return (
            f"this kernel is classified as `{primary}` from CycleTrace and mcTracer evidence; "
            "mcProfiler-dependent duty, cache, bank-conflict, IPC, and Roofline evidence is N/A."
        )
    if primary == "memory":
        return (
            f"this kernel is classified as `memory` because GVM/VLS/L2C/cache thresholds fired "
            f"(GVM share={cycle.get('gvm_pct', 0):.2f}%, L2C hit={fmt_pct(profiler.get('l2c_hit_rate_pct'))})."
        )
    if primary == "latency":
        return (
            f"this kernel is classified as `latency` because NOP/IPC evidence indicates pipeline bubbles "
            f"(NOP share={cycle.get('nop_pct', 0):.2f}%, real IPC={fmt_float(profiler.get('real_ipc'))})."
        )
    if primary == "occupancy":
        return (
            f"this kernel is classified as `occupancy` because launch occupancy is low "
            f"(effective occupancy={metrics['launch'].get('effective_occupancy_pct', 0):.2f}%)."
        )
    return (
        f"this kernel is classified as `{primary}` because compute-side evidence dominates "
        f"(MTE share={cycle.get('mte_pct', 0):.2f}%, AP MTE duty={fmt_pct(profiler.get('ap_mte_duty_pct'))}, "
        f"AP MMA duty={fmt_pct(profiler.get('ap_mma_duty_pct'))})."
    )

def mechanism_for_primary(metrics: dict[str, Any], dnoc_high_pct: float, wif_ratio: float) -> str:
    bound = metrics["bound"]
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    launch = metrics["launch"]
    primary = str(bound.get("primary", ""))
    if primary == "memory":
        return (
            f"memory/cache evidence is the current primary signal: GVM share={cycle.get('gvm_pct', 0):.2f}%, "
            f"VLS duty={fmt_pct(profiler.get('vls_duty_pct'))}, L2C duty={fmt_pct(profiler.get('l2c_duty_pct'))}, "
            f"L2C hit={fmt_pct(profiler.get('l2c_hit_rate_pct'))}, and DNOC >512-cycle share={dnoc_high_pct:.2f}%."
        )
    if primary == "latency":
        return (
            f"latency or issue-bubble evidence is the current primary signal: NOP share={cycle.get('nop_pct', 0):.2f}%, "
            f"real IPC={fmt_float(profiler.get('real_ipc'))}, and top ISU stall="
            f"{profiler.get('isu_stall_summary', {}).get('top', NA)} / {fmt_pct(profiler.get('isu_stall_summary', {}).get('top_pct'))}."
        )
    if primary == "occupancy":
        return (
            f"launch/parallelism evidence is the current primary signal: effective occupancy="
            f"{launch.get('effective_occupancy_pct', 0):.2f}%, achieved/dispatched waves="
            f"{fmt(profiler.get('achieved_waves'))}/{fmt(profiler.get('dispatched_waves'))}, and WIF P75/P25={wif_ratio:.2f}x."
        )
    return (
        f"compute-side evidence is the current primary signal: MTE share={cycle.get('mte_pct', 0):.2f}%, "
        f"MMA share={cycle.get('mma_pct', 0):.2f}%, AP MTE duty={fmt_pct(profiler.get('ap_mte_duty_pct'))}, "
        f"AP MMA duty={fmt_pct(profiler.get('ap_mma_duty_pct'))}, and MMA compute-cycle share="
        f"{fmt_pct(profiler.get('compute_instruction_cycles_summary', {}).get('mma_cycles_pct'))}."
    )

def weaker_hypotheses(metrics: dict[str, Any], dnoc_high_pct: float, wif_ratio: float) -> str:
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    launch = metrics["launch"]
    return (
        f"counter-evidence for alternate bounds is GVM share={cycle.get('gvm_pct', 0):.2f}%, "
        f"L2C hit={fmt_pct(profiler.get('l2c_hit_rate_pct'))}, DNOC >512-cycle share={dnoc_high_pct:.2f}%, "
        f"DPC imbalance={fmt_pct(profiler.get('dpc_compute_imbalance_pct'))}, WIF P75/P25={wif_ratio:.2f}x, "
        f"and effective occupancy={launch.get('effective_occupancy_pct', 0):.2f}%."
    )

def next_edit_lines(metrics: dict[str, Any]) -> list[str]:
    bound = metrics["bound"]
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    kernel = metrics["kernel"]
    primary = str(bound.get("primary", ""))
    file_line = "- File: kernel source file if it can be identified outside the current artifacts; current artifacts do not contain source-line attribution, so the exact file must be supplied or confirmed before editing."
    tensor_expected = expects_tensor_path(str(kernel.get("name", "")))
    if primary == "memory":
        if tensor_expected:
            change = "reduce the dominant traffic source shown by Memory Hierarchy, for example by improving tile reuse, removing repeated global loads/stores, or staging reused data through WSM when conflict risk is controlled."
            expected = "GVM/VLS pressure and global bytes should decrease, while L2C hit rate or achieved bandwidth context should improve without worsening shared-memory conflict."
        else:
            change = "reduce global read/write traffic for this non-tensor kernel, for example by removing redundant loads/stores, improving contiguous/vectorized access, or fusing adjacent elementwise work when that path exists."
            expected = "global bytes, GVM/VLS pressure, and DNOC requests should decrease without introducing shared-memory traffic or occupancy regressions."
        return [
            file_line,
            f"- Change: {change}",
            "- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare `cycle.gvm_pct`, `profiler.vls_duty_pct`, `profiler.l2c_hit_rate_pct`, DNOC latency buckets, global memory bytes, and `REPORT_<tag>.md`.",
            f"- Expected metric movement: {expected}",
        ]
    if primary == "latency":
        return [
            file_line,
            "- Change: shorten dependency chains and issue gaps by reordering load/compute, reducing unnecessary synchronization, or adding independent work between long-latency operations.",
            "- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare `cycle.nop_pct`, `profiler.real_ipc`, ISU stall summary, average wave life, and `REPORT_<tag>.md`.",
            "- Expected metric movement: NOP share and dominant ISU stall share should decrease, while real IPC should increase.",
        ]
    if primary == "occupancy":
        if tensor_expected:
            change = "reduce the limiting tensor-kernel resource after confirming whether registers, WSM tile footprint, or tile shape is the actual limiter."
        else:
            change = "adjust grid/block coverage for this non-tensor kernel first; reduce register or shared-memory footprint only if mcTracer/compiler evidence confirms a real resource limiter."
        return [
            file_line,
            f"- Change: {change}",
            "- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare effective occupancy, mtreg/shared-memory occupancy, achieved waves, WIF quartiles, DPC imbalance, and `REPORT_<tag>.md`.",
            "- Expected metric movement: effective occupancy and achieved waves should increase without creating new memory or bank-conflict pressure.",
        ]
    if (
        tensor_expected
        and cycle.get("mma_pct", 0) < 1
        and as_number(profiler.get("ap_mma_duty_pct")) < 3
    ):
        change = "confirm whether the profiled tensor-capable inner loop is actually lowered to the expected MMA instruction path for the same tile shape, then adjust the kernel implementation only if source or assembly evidence confirms a mismatch."
        expected = "AP MMA duty and CycleTrace MMA share should increase; if shared-memory efficiency remains below 80-85%, handle WSM layout padding/skew as the next separate edit."
    else:
        change = "focus the compute path indicated by SOL and compute-cycle split, reducing scalar/vector overhead or improving the intended compute instruction mix."
        expected = "the intended compute duty and compute-cycle share should increase without regressing memory hierarchy or occupancy metrics."
    return [
        file_line,
        f"- Change: {change}",
        "- Validation: rerun `trace_profile_pipeline.py run` on the same shape and compare AP duty ratios, CycleTrace instruction shares, compute instruction cycle split, shared memory efficiency, and `REPORT_<tag>.md`.",
        f"- Expected metric movement: {expected}",
    ]

def render_report(metrics: dict[str, Any]) -> str:
    tag = metrics["tag"]
    kernel = metrics["kernel"]
    launch = metrics["launch"]
    cycle = metrics["cycle"]
    profiler = metrics["profiler"]
    bound = metrics["bound"]
    coverage = metrics.get("metric_coverage", {})
    artifact_dir = metrics.get("artifact_dir", "")
    run_dir = render_run_dir(artifact_dir)
    has_profiler = bool(profiler.get("available", False))
    dnoc_total, dnoc_high, dnoc_high_pct = high_dnoc_share(profiler.get("dnoc_latency_histogram", {}))
    wif_p25 = cycle.get("wif_p25", 0)
    wif_p75 = cycle.get("wif_p75", 0)
    wif_ratio = ratio(wif_p75, wif_p25)
    private_reads = parse_number(profiler.get("memory_data_flow", {}).get("private_kernel_rd", 0))
    private_writes = parse_number(profiler.get("memory_data_flow", {}).get("private_kernel_wr", 0))
    atomic_keys = [
        key for key in profiler.get("all_scalars", {})
        if "atomic" in key.lower()
    ]
    atomic_total = sum(parse_number(profiler.get("all_scalars", {}).get(key, 0)) for key in atomic_keys)
    roof = profiler.get("roofline", {})
    cycle_trace_files = metrics.get("cycle_trace_files", [])
    cycle_dpc_id = dpc_id_for_rerun(metrics)
    multi_dpc_note = (
        f"- CycleTrace DPC selection: archived `{', '.join(cycle_trace_files)}`; "
        f"analysis uses primary `{metrics.get('cycle_trace_file', cycle_trace_filename())}`."
        if len(cycle_trace_files) > 1
        else ""
    )
    if expects_tensor_path(str(kernel.get("name", ""))):
        mma_interpretation = "Tensor/MMA path underuse signal for tensor-like kernels"
    else:
        mma_interpretation = "MMA path is not expected for this kernel type"

    lines = [
        f"# `{kernel.get('name', '')}` Trace Profiling Report",
        "",
        f"**Tag:** `{tag}`",
        "**Target GPU / arch:** MetaX C500",
        "**Tools:** CycleTrace + mcTracer" + (" + mcProfiler" if has_profiler else " (mcProfiler missing)"),
        f"**Run directory:** `{run_dir}`",
        f"**Artifact directory:** `{artifact_dir}`",
        "",
        "## 0. Profiling setup",
        "",
        f"- Kernel: `{kernel.get('name', '')}`",
        f"- Grid / block: `{kernel.get('grid', [])}` / `{kernel.get('block', [])}`",
        f"- CycleTrace JSON: `{artifact_dir}/{metrics.get('cycle_trace_file', cycle_trace_filename())}`",
        multi_dpc_note,
        f"- mcTracer JSON: `{artifact_dir}/tracer_out.json`",
        f"- mcProfiler JSON: `{artifact_dir}/mcprofiler_report_dumped.json`" if has_profiler else "- mcProfiler JSON: not collected",
        "- Bound mode: `" + str(bound.get("mode", "")) + "`",
        "",
        "Runnable pipeline:",
        "",
        "```bash",
        "python3 scripts/trace_report_env.py --config \"$ACTIVE_CONFIG\" run -- \\",
        "  python3 .trace-report/scripts/trace_profile_pipeline.py run \\",
        "  --source <profile-artifacts-dir> \\",
        "  --run-dir <profile-artifacts-dir> \\",
        f"  --tag {tag} \\",
        f"  --cycle-dpc-id {cycle_dpc_id} \\",
        f"  --bound-mode {bound.get('mode', 'coarse')}",
        "```",
        "",
        "## 1. Headline",
        "",
        f"- Bottleneck class: `{bound.get('type', '')}`",
        f"- Primary: `{bound.get('primary', '')}`",
        f"- Confidence: {confidence(metrics)}",
        f"- Dominant signal: {primary_signal(metrics)}",
        f"- One-line read: {one_line_read(metrics)}",
        "",
        "## 2. Evidence",
        "",
        "| Metric | Value | Source | Interpretation |",
        "|---|---:|---|---|",
        f"| Kernel span | {fmt(cycle.get('span_cycles', 0))} cycles | CycleTrace wave lifecycle | End-to-end trace span, lower is better |",
        f"| CycleTrace MTE share | {cycle.get('mte_pct', 0):.2f}% | CycleTrace instruction events | Vector compute path dominates instruction mix |",
        f"| CycleTrace MMA share | {cycle.get('mma_pct', 0):.2f}% | CycleTrace instruction events | {mma_interpretation} |",
        f"| AP MTE duty | {fmt_pct(profiler.get('ap_mte_duty_pct'))} | mcProfiler SOL | Hardware vector pipe duty |",
        f"| AP MMA duty | {fmt_pct(profiler.get('ap_mma_duty_pct'))} | mcProfiler SOL | Tensor pipe utilization |",
        f"| Shared memory efficiency | {fmt_pct(profiler.get('shared_memory_efficiency_pct'))} | mcProfiler WSM | Bank conflict signal when below 80-85% |",
        f"| Avg conflict cycles/inst | {fmt(profiler.get('avg_conflict_cycles_per_inst'))} | mcProfiler WSM | Conflict penalty per smem instruction |",
        f"| Effective occupancy bound | {launch.get('effective_occupancy_pct', 0):.2f}% | mcTracer | Lower of mtreg and shared-memory occupancy fields |",
        f"| L2C hit rate | {fmt_pct(profiler.get('l2c_hit_rate_pct'))} | mcProfiler memory hierarchy | High value weakens DRAM bandwidth hypothesis |",
        f"| WIF P75/P25 | {wif_ratio:.2f}x | CycleTrace wave lifecycle | Aggregate tail-effect proxy; >2x suggests imbalance |",
        f"| DPC compute imbalance | {fmt_pct(profiler.get('dpc_compute_imbalance_pct'))} | mcProfiler DPC cycles | Aggregate compute-balance signal |",
        f"| DNOC >512-cycle share | {dnoc_high_pct:.2f}% | mcProfiler DNOC histogram | DRAM latency-tail check |",
        f"| Roofline HBM usage | {fmt_pct(roof.get('hbm_usage_pct'))} | mcProfiler RoofLine | Achieved bandwidth divided by HBM peak from the RoofLine chart |",
        f"| Top ISU stall | {profiler.get('isu_stall_summary', {}).get('top', NA)} / {fmt_pct(profiler.get('isu_stall_summary', {}).get('top_pct'))} | mcProfiler ISU stall layout | Dominant issue-side stall bucket |",
        "",
        "## 3. Per-dimension analysis",
        "",
        "### 3.1 Occupancy & Launch",
        f"- Grid/block is `{kernel.get('grid', [])}` / `{kernel.get('block', [])}`; registers/thread={launch.get('registers_per_thread', 0)}, static shared={fmt(launch.get('static_shared_bytes', 0))} bytes.",
        f"- mcTracer occupancy fields are mtreg={launch.get('mtreg_occupancy_pct', 0):.2f}% and shared-memory={launch.get('shared_memory_occupancy_pct', 0):.2f}%; digest effective bound={launch.get('effective_occupancy_pct', 0):.2f}%.",
        f"- CycleTrace WIF mean/P25/P50/P75/max={cycle.get('wif_mean', 0):.2f}/{cycle.get('wif_p25', 0):.2f}/{cycle.get('wif_p50', 0):.2f}/{cycle.get('wif_p75', 0):.2f}/{cycle.get('wif_max', 0)} raw waves; P75/P25={wif_ratio:.2f}x.",
        f"- mcProfiler achieved/dispatched waves={fmt(profiler.get('achieved_waves'))}/{fmt(profiler.get('dispatched_waves'))}, workgroups={fmt(profiler.get('workgroups'))}, average wave life={fmt(profiler.get('average_wave_life_cycles'))} cycles.",
        f"- DPC compute balance: {dpc_balance_summary(profiler)}.",
        "",
        "### 3.2 Instruction Distribution",
        "- Primary classification uses CycleTrace `cat`, because `cat` identifies the C500 hardware category where the event executes.",
        "",
        *cycle_category_rows(cycle),
        "",
        f"- STE subevents by `name`: STE={cycle.get('raw_name_counts', {}).get('STE', 0):,}, S_NOP={cycle.get('s_nop', 0):,}, Branch={cycle.get('branch', 0):,}.",
        f"- GLOBAL subevents by `name`: GVM Load={cycle.get('gvm_load', 0):,}, GVM Store={cycle.get('gvm_store', 0):,}.",
        f"- ARRIVE subevents by `name`: Synchronization={cycle.get('raw_name_counts', {}).get('Synchronization', 0):,}.",
        "",
        "Hardware instruction cross-check:",
        "",
        *hardware_instruction_rows(cycle, profiler),
        "",
        "### 3.3 SOL / Hardware Duty",
        f"- AP MTE/STE/MMA duty={fmt_pct(profiler.get('ap_mte_duty_pct'))} / {fmt_pct(profiler.get('ap_ste_duty_pct'))} / {fmt_pct(profiler.get('ap_mma_duty_pct'))}.",
        f"- Real IPC={fmt_float(profiler.get('real_ipc'))}; instruction throughput={fmt(profiler.get('instruction_throughput'))}; throughput efficiency={fmt_pct(profiler.get('instruction_throughput_efficiency_pct'))}; compute-instruction busy duty={fmt_pct(profiler.get('compute_inst_busy_duty_pct'))}.",
        f"- Instructions per AP={fmt(profiler.get('instructions_per_ap'))}; average cycles/instruction={fmt_float(profiler.get('avg_cycles_per_instruction'))}; average all-stage latency/instruction={fmt_float(profiler.get('avg_latency_all_stages_per_instruction'))} cycles.",
        f"- Compute instruction cycle split: {compute_cycles_summary(profiler)}.",
        f"- ISU stall summary: {isu_stall_summary(profiler)}.",
        f"- AP active cycles={fmt(profiler.get('ap_active_cycles'))}; average AP busy cycles={fmt(profiler.get('average_ap_busy_cycles'))}; AP busy duty={fmt(profiler.get('ap_busy_duty_pct'))}. Treat AP busy duty as unit-sensitive context until the tool scale is confirmed.",
        "",
        "ISU stall cycle layout:",
        "",
        *stall_share_rows(profiler),
        "",
        "### 3.4 Memory Hierarchy",
        f"- VL1/L2C/SL1 hit rate={fmt_pct(profiler.get('vl1_hit_rate_pct'))} / {fmt_pct(profiler.get('l2c_hit_rate_pct'))} / {fmt_pct(profiler.get('sl1_hit_rate_pct'))}.",
        f"- DNOC read average latency={fmt(profiler.get('dnoc_read_average_latency'))} cycles; read/write requests={fmt(profiler.get('dnoc_read_req'))}/{fmt(profiler.get('dnoc_write_req'))}; achieved bandwidth={fmt_float(profiler.get('achieved_bandwidth_gbs'))} GB/s.",
        f"- Global memory bytes read/write={fmt(profiler.get('global_memory_read_bytes'))}/{fmt(profiler.get('global_memory_write_bytes'))}; global instructions read/write={fmt(profiler.get('global_read_instructions'))}/{fmt(profiler.get('global_write_instructions'))}.",
        f"- VL1 instructions read/write={fmt(profiler.get('vl1_read_instructions'))}/{fmt(profiler.get('vl1_write_instructions'))}; L2C instructions read/write={fmt(profiler.get('l2c_read_instructions'))}/{fmt(profiler.get('l2c_write_instructions'))}.",
        f"- Constant read path: total={fmt(profiler.get('constant_read_instructions'))}, SL1={fmt(profiler.get('constant_read_sl1_instructions'))}, L2={fmt(profiler.get('constant_read_l2_instructions'))}.",
        f"- DNOC latency histogram total={fmt(dnoc_total)} samples, >512-cycle samples={fmt(dnoc_high)} ({dnoc_high_pct:.2f}%).",
        f"- VL1 partition stalls: {vl1_partition_summary(profiler)}.",
        "",
        "DNOC latency buckets:",
        "",
        *dnoc_histogram_rows(profiler),
        "",
        "Memory data flow:",
        "",
        *memory_flow_summary(profiler.get("memory_data_flow", {})),
        "",
        "### 3.5 Bank Conflict & Private Memory",
        f"- Shared memory load/store instructions={fmt(profiler.get('shared_memory_load_instructions'))}/{fmt(profiler.get('shared_memory_store_instructions'))}.",
        f"- Avg cycles per load/store={fmt_float(profiler.get('avg_cycles_per_load'))}/{fmt_float(profiler.get('avg_cycles_per_store'))}; conflict cycles/inst={fmt(profiler.get('avg_conflict_cycles_per_inst'))}.",
        f"- Avg latency per load/store/atomic instruction={fmt_float(profiler.get('avg_latency_per_load_instruction'))}/{fmt_float(profiler.get('avg_latency_per_store_instruction'))}/{fmt_float(profiler.get('avg_latency_per_atomic_instruction'))} cycles; avg cycles per atomic instruction={fmt_float(profiler.get('avg_cycles_per_atomic_instruction'))}.",
        f"- Private memory from mcTracer: per_thread={launch.get('private_per_thread', 0)}, total={launch.get('private_total', 0)}.",
        f"- Private memory from mcProfiler memory flow: read={fmt(private_reads)}, write={fmt(private_writes)}.",
        f"- Atomic pressure: {fmt(atomic_total)} total atomic scalar counts across {len(atomic_keys)} atomic-related mcProfiler fields.",
        "",
        "### 3.6 Roofline",
        f"- Achieved point: {fmt_float(profiler.get('achieved_flops_tflops'))} TFLOP/s @ {fmt_float(profiler.get('achieved_bandwidth_gbs'))} GByte/s.",
        f"- Intensities: HBM={fmt_float(profiler.get('achieved_intensity_flop_per_byte'))}, VL1={fmt_float(roof.get('case_VL1_I', NA))}, L2C={fmt_float(roof.get('case_L2C_I', NA))} FLOP/Byte.",
        f"- Peak context from mcProfiler RoofLine: {roofline_peak_summary(roof)}.",
        f"- Bandwidth usage from mcProfiler RoofLine context: {roofline_usage_summary(roof)}.",
        f"- Roofline raw context: all_ops={fmt(roof.get('all_ops'))}, all_memacs={fmt(roof.get('all_memacs'))}, vl1_memacs={fmt(roof.get('vl1_memacs'))}, l2c_memacs={fmt(roof.get('l2c_memacs'))}, duration={fmt(roof.get('during'))}, core_clk={fmt_float(roof.get('core_clk'), 3)} GHz, mc_clk={fmt_float(roof.get('mc_clk'), 3)} GHz.",
        "- Roofline peak and usage fields are reported as mcProfiler chart context; operator-theoretical FLOP/Byte still requires external shape/formula metadata.",
        "",
        "### 3.7 Data Availability Boundaries",
        "- Current artifacts do not provide per-source-line or per-PC stall attribution; the report cannot name source-line hotspots.",
        "- Current artifacts do not provide PM-sampling or per-AP utilization time series; DPC cycles and WIF quartiles are aggregate balance proxies only.",
        "- Current artifacts do not expose a C500 sectors/request or useful-bytes/sector equivalent; memory flow and hit rates are not enough to claim global coalescing quality.",
        "",
        "### 3.8 Metric Coverage",
        "",
        *metric_coverage_rows(coverage),
        "",
        "## 4. Diagnosis",
        "",
    ]
    for item in metrics["diagnosis"]:
        lines.extend([
            f"### {item['severity'].upper()}: {item['title']}",
            "",
            f"- Evidence: {item['evidence']}",
            f"- Next: {item['next']}",
            "",
        ])

    lines.extend([
        "## 5. Inference Chain",
        "",
        f"1. Measured: {primary_signal(metrics)}.",
        f"2. Likely mechanism: {mechanism_for_primary(metrics, dnoc_high_pct, wif_ratio)}",
        f"3. Why other hypotheses are weaker: {weaker_hypotheses(metrics, dnoc_high_pct, wif_ratio)}",
        "4. Risk: the proposed kernel change can shift pressure across compute, memory hierarchy, WSM conflict, and occupancy; validate with the same shape before applying a second edit.",
        "",
        "## 6. Next Concrete Edit",
        "",
        *next_edit_lines(metrics),
        "",
        "## 7. Artifacts",
        "",
        f"- Full metrics: `analysis/metrics_all_{tag}.json`",
        f"- Key metrics: `analysis/metrics_key_{tag}.json`",
        f"- Digest: `analysis/digest_{tag}.md`",
        f"- This report: `REPORT_{tag}.md`",
        "",
        "## 8. Caveats",
        "",
        "- CycleTrace instruction `dur=4` is an issue-slot marker, not real execution latency.",
        "- GVM 600-cycle peak is a pressure proxy, not exact GVM buffer occupancy.",
        "- Effective occupancy uses mcTracer mtreg/shared-memory fields only; streg requires compiler/assembly evidence.",
        "- AP busy duty is reported as raw mcProfiler context because the example scale is ambiguous relative to per-pipe duty ratios.",
        "",
    ])
    return "\n".join(lines)
