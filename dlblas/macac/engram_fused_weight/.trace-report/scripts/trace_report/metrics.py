"""Metric assembly, bound classification, and diagnosis rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import find_artifact_dir, write_json
from .constants import KEY_METRICS, NA, cycle_trace_filename, primary_dpc_id, validate_dpc_id
from .parsers import parse_cycle_trace, parse_profiler, parse_tracer
from .render import render_digest, render_report
from .utils import (
    as_number,
    expects_tensor_path,
    fmt,
    fmt_pct,
    get_path,
    has_bank_conflict_signal,
    has_wsm_activity,
    parse_number,
)

def analyze_artifacts(
    run_dir: Path,
    tag: str,
    artifact_dir: Path | None = None,
    bound_mode: str = "coarse",
    cycle_dpc_id: str | int | None = None,
) -> dict[str, Any]:
    root = find_artifact_dir(run_dir, tag, artifact_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)

    requested_dpc_id = validate_dpc_id(cycle_dpc_id)
    cycle_file = cycle_trace_filename(requested_dpc_id)
    cycle_path = root / cycle_file
    tracer_path = root / "tracer_out.json"
    dumped_path = root / "mcprofiler_report_dumped.json"
    txt_path = root / "mcprofiler_report.txt.json"

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    launch = parse_tracer(tracer_path)
    cycle = parse_cycle_trace(cycle_path)
    profiler = parse_profiler(dumped_path if dumped_path.exists() else None, txt_path if txt_path.exists() else None)

    metrics = {
        "tag": tag,
        "artifact_dir": str(root),
        "cycle_dpc_id": requested_dpc_id,
        "cycle_dpc_primary": primary_dpc_id(requested_dpc_id),
        "cycle_trace_file": cycle_file,
        "cycle_trace_files": sorted(path.name for path in root.glob("c-trace_output_dpc_*.json")),
        "kernel": {
            "name": launch.get("kernel_name", ""),
            "grid": launch.get("grid", []),
            "block": launch.get("block", []),
            "span_cycles": cycle.get("span_cycles", 0),
        },
        "launch": launch,
        "cycle": cycle,
        "profiler": profiler,
        "bound": classify_bound(launch, cycle, profiler, bound_mode),
        "diagnosis": diagnose(launch, cycle, profiler),
    }
    metrics["metric_coverage"] = build_metric_coverage(metrics)
    key = {path: get_path(metrics, path) for path, _, _, _ in KEY_METRICS}

    write_json(analysis_dir / f"metrics_all_{tag}.json", metrics)
    write_json(
        analysis_dir / f"metrics_key_{tag}.json",
        {
            "tag": tag,
            "metrics": key,
            "metric_coverage_summary": metrics["metric_coverage"]["summary"],
        },
    )
    (analysis_dir / f"digest_{tag}.md").write_text(render_digest(metrics), encoding="utf-8")
    (run_dir / f"REPORT_{tag}.md").write_text(render_report(metrics), encoding="utf-8")
    return metrics

def classify_bound(
    launch: dict[str, Any],
    cycle: dict[str, Any],
    profiler: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Classify the dominant bound type from stable aggregate counters.

    coarse mode returns one of: compute, memory, latency, occupancy.
    detailed mode preserves compound C500-specific labels.
    """
    evidence = {
        "cycle_mma_pct": cycle.get("mma_pct", 0),
        "cycle_mte_pct": cycle.get("mte_pct", 0),
        "cycle_gvm_pct": cycle.get("gvm_pct", 0),
        "cycle_arrive_pct": cycle.get("arrive_pct", 0),
        "cycle_ldu_pct": cycle.get("ldu_pct", 0),
        "cycle_nop_pct": cycle.get("nop_pct", 0),
        "ap_mte_duty_pct": profiler.get("ap_mte_duty_pct", 0),
        "ap_ste_duty_pct": profiler.get("ap_ste_duty_pct", 0),
        "ap_mma_duty_pct": profiler.get("ap_mma_duty_pct", 0),
        "vls_duty_pct": profiler.get("vls_duty_pct", 0),
        "l2c_duty_pct": profiler.get("l2c_duty_pct", 0),
        "vl1_hit_rate_pct": profiler.get("vl1_hit_rate_pct", 0),
        "l2c_hit_rate_pct": profiler.get("l2c_hit_rate_pct", 0),
        "dnoc_read_average_latency": profiler.get("dnoc_read_average_latency", 0),
        "dnoc_read_req": profiler.get("dnoc_read_req", 0),
        "dnoc_write_req": profiler.get("dnoc_write_req", 0),
        "global_memory_read_bytes": profiler.get("global_memory_read_bytes", 0),
        "global_memory_write_bytes": profiler.get("global_memory_write_bytes", 0),
        "shared_memory_efficiency_pct": profiler.get("shared_memory_efficiency_pct", 0),
        "avg_conflict_cycles_per_inst": profiler.get("avg_conflict_cycles_per_inst", 0),
        "effective_occupancy_pct": launch.get("effective_occupancy_pct", 0),
        "mtreg_occupancy_pct": launch.get("mtreg_occupancy_pct", 0),
        "shared_memory_occupancy_pct": launch.get("shared_memory_occupancy_pct", 0),
        "real_ipc": profiler.get("real_ipc", 0),
        "achieved_bandwidth_gbs": profiler.get("achieved_bandwidth_gbs", 0),
        "roofline_hbm_usage_pct": profiler.get("roofline", {}).get("hbm_usage_pct", 0),
        "roofline_vl1_usage_pct": profiler.get("roofline", {}).get("vl1_usage_pct", 0),
        "roofline_l2c_usage_pct": profiler.get("roofline", {}).get("l2c_usage_pct", 0),
        "top_isu_stall": profiler.get("isu_stall_summary", {}).get("top", ""),
        "top_isu_stall_pct": profiler.get("isu_stall_summary", {}).get("top_pct", 0),
    }

    has_profiler = bool(profiler.get("available", False))
    tensor_expected = expects_tensor_path(str(launch.get("kernel_name", "")))
    evidence["tensor_path_expected"] = tensor_expected
    evidence["wsm_activity"] = has_wsm_activity(launch, cycle, profiler)
    is_tensor_underuse = tensor_expected and evidence["cycle_mma_pct"] < 1 and (
        has_profiler and as_number(evidence["ap_mma_duty_pct"]) < 3
    )
    is_mte_path = evidence["cycle_mte_pct"] > 30 or (
        has_profiler and as_number(evidence["ap_mte_duty_pct"]) > 15
    )
    is_bank_conflict = has_profiler and has_bank_conflict_signal(launch, cycle, profiler)
    effective_occupancy = as_number(evidence["effective_occupancy_pct"])
    is_low_occupancy = 0 < effective_occupancy < 25
    is_memory_pressure = (
        evidence["cycle_gvm_pct"] > 25
        or (has_profiler and as_number(evidence["vls_duty_pct"]) > 50)
        or (has_profiler and as_number(evidence["l2c_duty_pct"]) > 50)
        or (
            has_profiler
            and 0 < as_number(evidence["l2c_hit_rate_pct"]) < 80
            and (
                evidence["cycle_gvm_pct"] > 10
                or as_number(evidence["vls_duty_pct"]) > 10
                or as_number(evidence["l2c_duty_pct"]) > 10
            )
        )
    )
    is_latency_signal = evidence["cycle_nop_pct"] > 10 or (
        has_profiler and as_number(evidence["real_ipc"]) < 10 and not is_memory_pressure
    )
    is_issue_stall_signal = has_profiler and as_number(evidence["top_isu_stall_pct"]) > 50

    if mode == "detailed":
        labels: list[str] = []
        if is_tensor_underuse and is_mte_path:
            labels.append("mte/vector-compute-bound")
        elif has_profiler and as_number(evidence["ap_mma_duty_pct"]) >= 30:
            labels.append("mma/tensor-compute-bound")
        elif is_mte_path:
            labels.append("mte-compute-bound")
        if is_bank_conflict:
            labels.append("shared-memory-bank-conflict")
        if is_low_occupancy:
            labels.append("low-occupancy")
        if is_memory_pressure:
            labels.append("memory-bandwidth/traffic-pressure")
        if is_latency_signal:
            labels.append("latency/pipeline-bubble")
        if not labels:
            labels.append("unclear")
        primary = labels[0]
        bound_type = " + ".join(labels)
    else:
        if is_low_occupancy and not (is_tensor_underuse or is_mte_path):
            primary = "occupancy"
        elif is_memory_pressure:
            primary = "memory"
        elif is_latency_signal and not (is_tensor_underuse or is_mte_path):
            primary = "latency"
        else:
            primary = "compute"
        bound_type = primary

    rationale = []
    if is_tensor_underuse:
        rationale.append("MMA share and AP MMA duty are very low, so this is not tensor-core saturation.")
    if is_mte_path:
        rationale.append("MTE instruction share / duty dominates the executed compute path.")
    if is_bank_conflict:
        rationale.append("Shared-memory efficiency and conflict cycles indicate WSM bank-conflict pressure.")
    if is_low_occupancy:
        rationale.append("mcTracer occupancy bound is below 25%; streg is unavailable in current JSON.")
    if is_memory_pressure:
        rationale.append("Memory pressure threshold fired from GVM/VLS/L2C/cache evidence.")
    elif has_profiler:
        rationale.append("L2C hit rate and achieved bandwidth do not indicate global-memory bandwidth saturation.")
    if not has_profiler:
        rationale.append("mcProfiler artifacts are missing; profiler duty, IPC, cache, bank-conflict, and Roofline evidence are N/A.")
    if is_latency_signal:
        rationale.append("NOP/IPC thresholds indicate latency or pipeline-bubble pressure.")
    if is_issue_stall_signal:
        rationale.append("ISU stall layout has a dominant stall bucket; use it as supporting issue-side evidence, not an NCU-style warp stall reason.")

    return {
        "mode": mode,
        "type": bound_type,
        "primary": primary,
        "evidence": evidence,
        "rationale": rationale,
        "thresholds": {
            "tensor_underuse": "tensor-like kernel and cycle_mma_pct < 1 and ap_mma_duty_pct < 3",
            "mte_path": "cycle_mte_pct > 30 or ap_mte_duty_pct > 15",
            "bank_conflict": "avg_conflict_cycles_per_inst > 0.5, or WSM activity with 0 < shared_memory_efficiency_pct < 80",
            "low_occupancy": "0 < effective_occupancy_pct < 25",
            "memory_pressure": "gvm_pct > 25, vls/l2c duty > 50, or l2c_hit_rate_pct < 80 with supporting GVM/VLS/L2C activity",
            "latency_signal": "nop_pct > 10 or real_ipc < 10 when memory pressure is absent",
        },
    }

def diagnose(launch: dict[str, Any], cycle: dict[str, Any], profiler: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    has_profiler = bool(profiler.get("available", False))
    tensor_expected = expects_tensor_path(str(launch.get("kernel_name", "")))
    if (
        has_profiler
        and tensor_expected
        and as_number(profiler.get("ap_mma_duty_pct")) < 3
        and cycle.get("mma_pct", 0) < 1
    ):
        findings.append({
            "severity": "high",
            "title": "Tensor core underuse",
            "evidence": f"MMA share={cycle.get('mma_pct', 0):.2f}%, AP MMA duty={fmt_pct(profiler.get('ap_mma_duty_pct'))}",
            "next": "Check whether the kernel maps the tensor-capable inner loop to the expected MMA instruction path instead of vector MTE work; require source or assembly evidence before naming an implementation change.",
        })
    if has_profiler and has_bank_conflict_signal(launch, cycle, profiler):
        findings.append({
            "severity": "high",
            "title": "Shared memory bank conflict pressure",
            "evidence": f"SMEM efficiency={fmt_pct(profiler.get('shared_memory_efficiency_pct'))}, conflict cycles/inst={fmt(profiler.get('avg_conflict_cycles_per_inst'))}",
            "next": "Inspect WSM layout, vectorized access stride, and padding/skew options.",
        })
    effective_occupancy = as_number(launch.get("effective_occupancy_pct", 0))
    if 0 < effective_occupancy < 25:
        if tensor_expected:
            next_action = "For this tensor-path kernel, inspect register pressure, shared-memory tile footprint, and tile shape before changing launch geometry."
        else:
            next_action = "For this non-tensor kernel, inspect grid/block coverage and only then reduce resource footprint if mcTracer shows a real limiter."
        findings.append({
            "severity": "medium",
            "title": "Low launch occupancy bound",
            "evidence": f"mtreg={launch.get('mtreg_occupancy_pct', 0):.2f}%, smem={launch.get('shared_memory_occupancy_pct', 0):.2f}%",
            "next": f"{next_action} streg limit still needs compiler/assembly evidence.",
        })
    elif effective_occupancy == 0 and (
        launch.get("mtreg_occupancy_pct", 0) == 0
        or launch.get("shared_memory_occupancy_pct", 0) == 0
    ):
        if tensor_expected:
            next_action = "Treat the zero occupancy fields as unavailable until mcTracer or compiler resource output confirms the tensor tile resource limiter."
        else:
            next_action = "Treat the zero occupancy fields as unavailable for this non-tensor kernel; do not reduce resources based on this value alone."
        findings.append({
            "severity": "info",
            "title": "Occupancy fields unavailable or inconsistent",
            "evidence": f"mtreg={launch.get('mtreg_occupancy_pct', 0):.2f}%, smem={launch.get('shared_memory_occupancy_pct', 0):.2f}%",
            "next": next_action,
        })
    if cycle.get("nop_pct", 0) > 10:
        findings.append({
            "severity": "medium",
            "title": "Pipeline bubble signal",
            "evidence": f"NOP share={cycle.get('nop_pct', 0):.2f}%",
            "next": "Correlate with barriers, load-use distance, and mcProfiler IPC.",
        })
    if not findings:
        findings.append({
            "severity": "info",
            "title": "No high-confidence rule fired",
            "evidence": "Rule thresholds did not identify a dominant issue.",
            "next": "Compare against an optimized version and inspect per-source behavior.",
        })
    return findings

def build_metric_coverage(metrics: dict[str, Any]) -> dict[str, Any]:
    launch = metrics.get("launch", {})
    cycle = metrics.get("cycle", {})
    profiler = metrics.get("profiler", {})
    flow = profiler.get("memory_data_flow", {})
    atomic_keys = [key for key in profiler.get("all_scalars", {}) if "atomic" in key.lower()]

    report_metric_groups = [
        {
            "group": "kernel_identity",
            "status": "reported",
            "fields": ["kernel.name", "kernel.grid", "kernel.block", "artifact_dir"],
            "role": "Reproducibility and target-kernel confirmation.",
        },
        {
            "group": "bound_classification",
            "status": "reported",
            "fields": ["bound.mode", "bound.type", "bound.primary", "bound.evidence", "bound.rationale"],
            "role": "Headline diagnosis and machine-readable decision evidence.",
        },
        {
            "group": "cycletrace_instruction_mix",
            "status": "reported",
            "fields": ["cycle.category_counts", "cycle.category_pcts", "cycle.mte_pct", "cycle.ste_pct", "cycle.mma_pct", "cycle.bsm_pct", "cycle.gvm_pct", "cycle.arrive_pct", "cycle.ldu_pct", "cycle.sync_count"],
            "role": "Hardware-category instruction mix from CycleTrace `cat`; name-level NOP/Branch/GVM load-store are subevent context.",
        },
        {
            "group": "launch_resources",
            "status": "reported",
            "fields": ["launch.registers_per_thread", "launch.static_shared_bytes", "launch.dynamic_shared_bytes", "launch.private_per_thread", "launch.mtreg_occupancy_pct", "launch.shared_memory_occupancy_pct"],
            "role": "Occupancy and resource-limit diagnosis.",
        },
        {
            "group": "tail_and_balance",
            "status": "reported",
            "fields": ["cycle.wif_p25", "cycle.wif_p50", "cycle.wif_p75", "cycle.wif_max", "profiler.dpc_compute_imbalance_pct", "profiler.achieved_waves", "profiler.dispatched_waves"],
            "role": "Aggregate tail-effect and DPC balance proxy.",
        },
        {
            "group": "hardware_duty",
            "status": "reported",
            "fields": ["profiler.ap_mte_duty_pct", "profiler.ap_ste_duty_pct", "profiler.ap_mma_duty_pct", "profiler.real_ipc", "profiler.instruction_throughput", "profiler.instruction_throughput_efficiency_pct", "profiler.compute_instruction_cycles_summary", "profiler.isu_stall_summary"],
            "role": "Hardware utilization, instruction efficiency, compute-cycle split, and issue-side stall context.",
        },
        {
            "group": "memory_hierarchy",
            "status": "reported",
            "fields": ["profiler.vl1_hit_rate_pct", "profiler.l2c_hit_rate_pct", "profiler.sl1_hit_rate_pct", "profiler.dnoc_read_average_latency", "profiler.dnoc_read_req", "profiler.dnoc_write_req", "profiler.global_memory_read_bytes", "profiler.global_memory_write_bytes", "profiler.memory_data_flow", "profiler.dnoc_latency_histogram_pct", "profiler.vl1_partition_stall_summary"],
            "role": "Memory-bound, traffic volume, latency-tail, and cache-pressure checks.",
        },
        {
            "group": "bank_conflict_private_atomic",
            "status": "reported",
            "fields": ["profiler.shared_memory_efficiency_pct", "profiler.avg_conflict_cycles_per_inst", "profiler.avg_cycles_per_load", "profiler.avg_cycles_per_store", "launch.private_per_thread", "profiler.memory_data_flow.private_kernel_rd"],
            "role": "WSM bank-conflict, spill/private-memory, and serialization context.",
        },
        {
            "group": "roofline",
            "status": "reported",
            "fields": ["profiler.achieved_flops_tflops", "profiler.achieved_bandwidth_gbs", "profiler.achieved_intensity_flop_per_byte", "profiler.roofline.case_VL1_I", "profiler.roofline.case_L2C_I", "profiler.roofline.hbm_usage_pct", "profiler.roofline.vl1_usage_pct", "profiler.roofline.l2c_usage_pct", "profiler.roofline.peak_fma_tflops", "profiler.roofline.peak_mma_fp16_tflops"],
            "role": "Empirical compute/memory placement plus mcProfiler RoofLine chart peak and usage context.",
        },
    ]

    parsed_not_promoted = [
        {
            "field": "cycle.raw_name_counts / cycle.raw_cat_counts",
            "role": "Raw event audit.",
            "reason": "Duplicates summarized hardware-category and subevent counts; mainly useful for parser validation.",
        },
        {
            "field": "cycle.issue_density_inst_per_cycle",
            "role": "CycleTrace issue-density sanity check.",
            "reason": "Wave lifecycle span is not equivalent to real IPC; mcProfiler IPC is more diagnostic.",
        },
        {
            "field": "launch.kernel_event_duration",
            "role": "Host trace duration context.",
            "reason": "Device-kernel report prioritizes CycleTrace span and mcProfiler cycles.",
            "value": launch.get("kernel_event_duration", 0),
        },
        {
            "field": "launch.top_api_calls_by_duration",
            "role": "Runtime/API overhead context.",
            "reason": "Not part of the current device-kernel bottleneck diagnosis.",
            "available_count": len(launch.get("top_api_calls_by_duration", [])),
        },
        {
            "field": "profiler.all_scalars / profiler.all_charts",
            "role": "Lossless mcProfiler raw evidence.",
            "reason": "Too verbose for the report body; retained for audit and future parser expansion.",
            "available_scalars": len(profiler.get("all_scalars", {})),
            "available_charts": len(profiler.get("all_charts", {})),
        },
        {
            "field": "profiler.memory_data_flow raw entries",
            "role": "Detailed path-level memory flow.",
            "reason": "Report shows recognized read/write groups; raw entries remain available for path-specific investigations.",
            "available_count": len(flow),
        },
        {
            "field": "profiler.roofline raw entries",
            "role": "Raw mcProfiler RoofLine chart inputs.",
            "reason": "Report promotes peak, usage, and intensity context; raw fields remain available for audit.",
            "available_count": len(profiler.get("roofline", {})),
        },
        {
            "field": "atomic-related mcProfiler scalar fields",
            "role": "Atomic/serialization pressure.",
            "reason": "Report summarizes total atomic pressure; individual counters are retained in metrics_all.",
            "available_count": len(atomic_keys),
        },
    ]

    unavailable = [
        {
            "dimension": "source-line/PC/SASS/PTX attribution",
            "impact": "Cannot name exact source-line hotspots or confirm instruction mnemonics.",
            "required_artifact": "Compiler/source instrumentation or disassembly.",
        },
        {
            "dimension": "PM-sampling or per-AP timeline",
            "impact": "Cannot prove timeline shape, alternating idle phases, or late low occupancy.",
            "required_artifact": "PM sampling or richer CycleTrace timeline export.",
        },
        {
            "dimension": "sector/request and useful-bytes/sector counters",
            "impact": "Cannot claim global-memory coalescing quality.",
            "required_artifact": "C500 equivalent of sector/request or byte-utilization counters.",
        },
        {
            "dimension": "complete occupancy limiter decomposition",
            "impact": "streg and block-limit causes cannot be separated from current mtreg/shared-memory fields.",
            "required_artifact": "Compiler/assembly resource report or richer mcTracer occupancy fields.",
        },
        {
            "dimension": "exact GVM/BSM buffer occupancy over time",
            "impact": "GVM 600-cycle peak remains a pressure proxy, not precise buffer occupancy.",
            "required_artifact": "CycleTrace Web UI derived views or richer trace export.",
        },
    ]

    return {
        "summary": {
            "reported_metric_groups": len(report_metric_groups),
            "parsed_not_promoted_groups": len(parsed_not_promoted),
            "unavailable_dimensions": len(unavailable),
        },
        "reported_metric_groups": report_metric_groups,
        "parsed_not_promoted": parsed_not_promoted,
        "unavailable_dimensions": unavailable,
    }
