"""Parsers for CycleTrace, mcTracer, and mcProfiler artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .artifacts import load_json
from .constants import NA
from .utils import pct, quantile, sliding_window_peak, parse_number, parse_optional_number

C500_AP_COUNT = 104
C500_VL1_BYTES_PER_CYCLE_PER_AP = 128
C500_L2C_BYTES_PER_CYCLE = 4096

def parse_cycle_trace(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    events = raw.get("traceEvents", [])
    if not isinstance(events, list) or not events:
        raise ValueError(f"CycleTrace JSON has no traceEvents: {path}")
    counts = Counter(e.get("name", "") for e in events)
    cats = Counter(e.get("cat", "") for e in events)

    hardware_cats = ("MTE", "STE", "MMA", "BSM", "GLOBAL", "ARRIVE", "LDU")
    total = sum(cats[name] for name in hardware_cats)
    if total <= 0:
        raise ValueError(f"CycleTrace JSON has no hardware instruction events: {path}")

    starts = sorted(e.get("ts", 0) for e in events if e.get("name") == "Wave start")
    ends = sorted(e.get("ts", 0) for e in events if e.get("name") == "Wave end")
    span = (max(ends) - min(starts)) if starts and ends else 0

    sweep: list[tuple[int, int]] = [(ts, 1) for ts in starts] + [(ts, -1) for ts in ends]
    sweep.sort()
    active = 0
    samples: list[int] = []
    for _, delta in sweep:
        active += delta
        samples.append(active)

    gvm_ts = sorted(
        e.get("ts", 0)
        for e in events
        if e.get("name") in ("GVM Load", "GVM Store")
    )
    gvm_peak = sliding_window_peak(gvm_ts, 600)

    return {
        "source": str(path),
        "event_count": len(events),
        "raw_name_counts": dict(counts),
        "raw_cat_counts": dict(cats),
        "total_instructions": total,
        "category_counts": {name: cats[name] for name in hardware_cats},
        "category_pcts": {name: pct(cats[name], total) for name in hardware_cats},
        "category_notes": {
            "basis": "CycleTrace `cat` is the primary instruction-distribution key because it identifies the C500 hardware category where the event executes.",
            "STE": "`STE` category includes scalar STE events plus STE-category S_NOP and Branch event names.",
            "GLOBAL": "`GLOBAL` category covers GVM Load and GVM Store event names.",
            "ARRIVE": "`ARRIVE` category covers Synchronization event names.",
        },
        "mte": cats["MTE"],
        "ste": cats["STE"],
        "s_nop": counts["S_NOP"],
        "branch": counts["Branch"],
        "mma": cats["MMA"],
        "bsm": cats["BSM"],
        "gvm_load": counts["GVM Load"],
        "gvm_store": counts["GVM Store"],
        "gvm_total": cats["GLOBAL"],
        "arrive": cats["ARRIVE"],
        "ldu": cats["LDU"],
        "wave_start": counts["Wave start"],
        "wave_end": counts["Wave end"],
        "mma_pct": pct(cats["MMA"], total),
        "mte_pct": pct(cats["MTE"], total),
        "ste_pct": pct(cats["STE"], total),
        "bsm_pct": pct(cats["BSM"], total),
        "gvm_pct": pct(cats["GLOBAL"], total),
        "arrive_pct": pct(cats["ARRIVE"], total),
        "ldu_pct": pct(cats["LDU"], total),
        "nop_pct": pct(counts["S_NOP"], total),
        "branch_pct": pct(counts["Branch"], total),
        "sync_count": counts["Synchronization"],
        "span_cycles": span,
        "issue_density_inst_per_cycle": round(total / span, 6) if span else 0.0,
        "wif_mean": round(sum(samples) / len(samples), 4) if samples else 0.0,
        "wif_max": max(samples) if samples else 0,
        "wif_p25": round(quantile(samples, 0.25), 4),
        "wif_p50": round(quantile(samples, 0.50), 4),
        "wif_p75": round(quantile(samples, 0.75), 4),
        "gvm_600cy_peak": gvm_peak,
        "notes": {
            "cycletrace_duration": "Instruction dur=4 is treated as issue slot width, not real execution latency.",
            "gvm_600cy_peak": "Max GVM load/store issue count in a 600-cycle sliding window; pressure proxy, not exact buffer occupancy.",
            "wif_scope": "Raw CycleTrace wave count distribution. Confirm counter scope before converting to waves/AP.",
        },
    }

def parse_tracer(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    events = raw.get("traceEvents", [])
    kernel_events = [
        e for e in events
        if e.get("ph") == "X"
        and isinstance(e.get("args"), dict)
        and {"grid", "block", "mem"}.issubset(e["args"])
    ]
    kernel = kernel_events[0] if kernel_events else {}
    args = kernel.get("args", {})
    grid = args.get("grid", {})
    block = args.get("block", {})
    mem = args.get("mem", {})
    mtreg_occ = float(args.get("mtreg_occupancy(%)", 0) or 0)
    smem_occ = float(args.get("shared_memeory_occupancy(%)", 0) or 0)

    api_calls = [
        {"name": e.get("name", ""), "dur": e.get("dur", 0)}
        for e in events
        if e.get("ph") == "X" and e.get("pid") == 1
    ]
    top_api = sorted(api_calls, key=lambda x: x["dur"], reverse=True)[:10]

    return {
        "source": str(path),
        "kernel_name": kernel.get("name", ""),
        "mangled_name": args.get("name", ""),
        "grid": [grid.get("x", 0), grid.get("y", 0), grid.get("z", 0)],
        "block": [block.get("x", 0), block.get("y", 0), block.get("z", 0)],
        "registers_per_thread": int(mem.get("registers_per_thread", 0) or 0),
        "static_shared_bytes": int(mem.get("static_shared", 0) or 0),
        "dynamic_shared_bytes": int(mem.get("dynamic_shared", 0) or 0),
        "private_per_thread": int(mem.get("private_per_thread", 0) or 0),
        "private_total": int(mem.get("private_total", 0) or 0),
        "mtreg_occupancy_pct": mtreg_occ,
        "shared_memory_occupancy_pct": smem_occ,
        "effective_occupancy_pct": min(x for x in (mtreg_occ, smem_occ) if x > 0) if mtreg_occ or smem_occ else 0.0,
        "kernel_event_duration": kernel.get("dur", 0),
        "top_api_calls_by_duration": top_api,
        "notes": {
            "effective_occupancy_pct": "Computed only from mcTracer mtreg/shared-memory fields. streg limiter is not available in this JSON.",
        },
    }

def parse_profiler(dumped_path: Path | None, txt_path: Path | None) -> dict[str, Any]:
    dumped = load_json(dumped_path) if dumped_path and dumped_path.exists() else {}
    txt = load_json(txt_path) if txt_path and txt_path.exists() else {}
    if not dumped and not txt:
        return empty_profiler()
    sections = dumped or txt
    fmt = "dumped" if dumped else "txt"

    scalars: dict[str, Any] = {}
    charts: dict[str, dict[str, Any]] = {}
    for section, entries in sections.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            name = entry.get("name", "")
            value = entry.get("data") if fmt == "dumped" else entry.get("value")
            if isinstance(value, dict):
                data = value.get("data", value) if fmt == "txt" else value
                charts[name] = data if isinstance(data, dict) else {}
            else:
                scalars[name] = parse_optional_number(value)

    if txt:
        roof = extract_roofline(txt)
    else:
        roof = extract_roofline(dumped)
    roof = {key: parse_optional_number(value) for key, value in roof.items()}

    instr = charts.get("Instructions Comparison", {})
    dpc_cycles = {str(k): parse_number(v) for k, v in charts.get("DPC Computing cycles comparison", {}).items()}
    dpc_summary = min_avg_max_spread(dpc_cycles)
    hw_total = sum(parse_number(v) for v in instr.values())
    scalar = lambda name: scalars.get(name, NA)
    dispatched_waves = scalars.get("Dispatched waves", scalars.get("WAVES", NA))
    latency_histogram = charts.get("Dnoc Read Latency Histogram", {})
    vl1_partition_stalls = charts.get("VL1 partition stall cycles layout", {})
    compute_cycles = charts.get("Compute Instructions Cycles Rate", {})
    isu_stalls = charts.get("ISU stall cycles layout", {})
    flow = charts.get("Memory Data Flow Chart", {})

    roof.update(derive_roofline_metrics(roof))

    out = {
        "available": True,
        "source_dumped": str(dumped_path) if dumped_path else "",
        "source_txt": str(txt_path) if txt_path else "",
        "total_instructions": scalar("Total Instructions"),
        "compute_instructions": scalar("Compute Instructions"),
        "memory_instructions": scalar("Memory Instructions"),
        "total_cycles": scalar("Total Cycles"),
        "average_ap_busy_cycles": scalar("Average AP busy cycles"),
        "ap_busy_duty_pct": scalar("AP busy Duty"),
        "compute_inst_busy_duty_pct": scalar("Compute Instructions busy Duty"),
        "workgroups": scalar("WORKGROUPS"),
        "waves": scalar("WAVES"),
        "average_wave_life_cycles": scalar("Average Wave life cycles"),
        "ap_active_cycles": scalar("AP active cycles"),
        "ap_mte_duty_pct": scalar("AP MTE Duty ratio"),
        "ap_ste_duty_pct": scalar("AP STE Duty ratio"),
        "ap_mma_duty_pct": scalar("AP MMA Duty ratio"),
        "vls_duty_pct": scalar("VLS Duty ratio"),
        "l2c_duty_pct": scalar("L2C Duty ratio"),
        "real_ipc": scalar("instruction per cycle"),
        "instruction_throughput": scalar("instruction throughput"),
        "instruction_throughput_efficiency_pct": scalar("instruction throughput efficiency"),
        "instructions_per_ap": scalar("instructions per AP"),
        "avg_cycles_per_instruction": scalar("average cycles per instruction"),
        "avg_latency_all_stages_per_instruction": scalar("average latency of all stages per instruction"),
        "avg_latency_per_load_instruction": scalar("average latency per load instruction"),
        "avg_latency_per_store_instruction": scalar("average latency per store instruction"),
        "avg_latency_per_atomic_instruction": scalar("average latency per atomic instruction"),
        "avg_cycles_per_atomic_instruction": scalar("average cycles per atomic instruction"),
        "vl1_hit_rate_pct": scalar("VL1 Hit Rate"),
        "l2c_hit_rate_pct": scalar("L2C Hit Rate"),
        "sl1_hit_rate_pct": scalar("Constant SL1 Hit Rate"),
        "dnoc_read_average_latency": scalar("Dnoc Read Average Latency"),
        "dnoc_read_req": scalar("Dnoc Read Req"),
        "dnoc_write_req": scalar("Dnoc Write Req"),
        "memory_access_per_second": scalar("Memory Access per Second"),
        "global_memory_read_bytes": scalar("Global Memory Read bytes"),
        "global_memory_write_bytes": scalar("Global Memory Write bytes"),
        "global_read_instructions": scalar("Global Read Instructions"),
        "global_write_instructions": scalar("Global Write Instructions"),
        "vl1_read_instructions": scalar("VL1 Read Instructions"),
        "vl1_write_instructions": scalar("VL1 Write Instructions"),
        "l2c_read_instructions": scalar("L2C Read Instructions"),
        "l2c_write_instructions": scalar("L2C Write Instructions"),
        "constant_read_instructions": scalar("Constant Read Instructions"),
        "constant_read_sl1_instructions": scalar("Constant Read SL1 Instructions"),
        "constant_read_l2_instructions": scalar("Constant Read L2 Instructions"),
        "shared_memory_load_instructions": scalar("load instructions"),
        "shared_memory_store_instructions": scalar("store instructions"),
        "shared_memory_efficiency_pct": scalar("shared memory access efficiency"),
        "avg_conflict_cycles_per_inst": scalar("average conflict cycles per instruction"),
        "avg_cycles_per_load": scalar("average cycles per load instruction"),
        "avg_cycles_per_store": scalar("average cycles per store instruction"),
        "achieved_waves": scalar("Achieved waves"),
        "dispatched_waves": dispatched_waves,
        "hardware_instruction_counts": instr,
        "hardware_instruction_total": hw_total,
        "dpc_compute_cycles": dpc_cycles,
        "dpc_compute_summary": dpc_summary,
        "dpc_compute_imbalance_pct": dpc_summary["spread_pct"],
        "dnoc_latency_histogram": latency_histogram,
        "dnoc_latency_histogram_pct": bucket_percentages(latency_histogram),
        "vl1_partition_stalls": vl1_partition_stalls,
        "vl1_partition_stall_summary": min_avg_max_spread(vl1_partition_stalls),
        "memory_data_flow": flow,
        "compute_instruction_cycles_rate": compute_cycles,
        "compute_instruction_cycles_summary": compute_cycles_summary(compute_cycles),
        "isu_stall_cycles": isu_stalls,
        "isu_stall_summary": stall_summary(isu_stalls),
        "roofline": roof,
        "all_scalars": scalars,
        "all_charts": charts,
    }
    for name, value in instr.items():
        out[f"hw_{name.lower()}_pct"] = pct(parse_number(value), hw_total)
    out["achieved_flops_tflops"] = roof.get("case_flops", NA)
    out["achieved_bandwidth_gbs"] = roof.get("case_bandwith", NA)
    out["achieved_intensity_flop_per_byte"] = roof.get("case_I", NA)
    return out

def empty_profiler() -> dict[str, Any]:
    scalar_keys = (
        "total_instructions",
        "compute_instructions",
        "memory_instructions",
        "total_cycles",
        "average_ap_busy_cycles",
        "ap_busy_duty_pct",
        "compute_inst_busy_duty_pct",
        "workgroups",
        "waves",
        "average_wave_life_cycles",
        "ap_active_cycles",
        "ap_mte_duty_pct",
        "ap_ste_duty_pct",
        "ap_mma_duty_pct",
        "vls_duty_pct",
        "l2c_duty_pct",
        "real_ipc",
        "instruction_throughput",
        "instruction_throughput_efficiency_pct",
        "instructions_per_ap",
        "avg_cycles_per_instruction",
        "avg_latency_all_stages_per_instruction",
        "avg_latency_per_load_instruction",
        "avg_latency_per_store_instruction",
        "avg_latency_per_atomic_instruction",
        "avg_cycles_per_atomic_instruction",
        "vl1_hit_rate_pct",
        "l2c_hit_rate_pct",
        "sl1_hit_rate_pct",
        "dnoc_read_average_latency",
        "dnoc_read_req",
        "dnoc_write_req",
        "memory_access_per_second",
        "global_memory_read_bytes",
        "global_memory_write_bytes",
        "global_read_instructions",
        "global_write_instructions",
        "vl1_read_instructions",
        "vl1_write_instructions",
        "l2c_read_instructions",
        "l2c_write_instructions",
        "constant_read_instructions",
        "constant_read_sl1_instructions",
        "constant_read_l2_instructions",
        "shared_memory_load_instructions",
        "shared_memory_store_instructions",
        "shared_memory_efficiency_pct",
        "avg_conflict_cycles_per_inst",
        "avg_cycles_per_load",
        "avg_cycles_per_store",
        "achieved_waves",
        "dispatched_waves",
        "hardware_instruction_total",
        "dpc_compute_imbalance_pct",
        "achieved_flops_tflops",
        "achieved_bandwidth_gbs",
        "achieved_intensity_flop_per_byte",
    )
    out: dict[str, Any] = {key: NA for key in scalar_keys}
    out.update({
        "available": False,
        "source_dumped": "",
        "source_txt": "",
        "hardware_instruction_counts": {},
        "dpc_compute_cycles": {},
        "dpc_compute_summary": {},
        "dnoc_latency_histogram": {},
        "dnoc_latency_histogram_pct": {},
        "vl1_partition_stalls": {},
        "vl1_partition_stall_summary": {},
        "memory_data_flow": {},
        "compute_instruction_cycles_rate": {},
        "compute_instruction_cycles_summary": {},
        "isu_stall_cycles": {},
        "isu_stall_summary": {},
        "roofline": {},
        "all_scalars": {},
        "all_charts": {},
    })
    return out

def extract_roofline(raw: dict[str, Any]) -> dict[str, Any]:
    for entry in raw.get("Internal", []):
        if entry.get("name") != "RoofLine":
            continue
        value = entry.get("value", {})
        if isinstance(value, dict):
            data = value.get("data", {})
            processed = value.get("processed_data", {})
            return {**processed, **data}
        data = entry.get("data", {})
        processed = entry.get("processed_data", {})
        if isinstance(data, dict) or isinstance(processed, dict):
            return {**(processed or {}), **(data or {})}
    return {}

def min_avg_max_spread(values: dict[str, Any]) -> dict[str, float | str]:
    nums = [parse_number(v) for v in values.values()]
    if not nums:
        return {"min": NA, "avg": NA, "max": NA, "spread_pct": NA}
    avg = sum(nums) / len(nums)
    return {
        "min": min(nums),
        "avg": avg,
        "max": max(nums),
        "spread_pct": pct(max(nums) - min(nums), avg),
    }

def bucket_percentages(values: dict[str, Any]) -> dict[str, float]:
    total = sum(parse_number(v) for v in values.values())
    return {str(k): pct(parse_number(v), total) for k, v in values.items()}

def stall_summary(values: dict[str, Any]) -> dict[str, Any]:
    nums = {str(k): parse_number(v) for k, v in values.items()}
    total = sum(nums.values())
    top_name, top_value = ("", 0.0)
    if nums:
        top_name, top_value = max(nums.items(), key=lambda item: item[1])
    return {
        "total": total,
        "top": top_name,
        "top_cycles": top_value,
        "top_pct": pct(top_value, total),
        "shares_pct": {name: pct(value, total) for name, value in nums.items()},
    }

def compute_cycles_summary(values: dict[str, Any]) -> dict[str, Any]:
    nums = {str(k): parse_number(v) for k, v in values.items()}
    total = sum(nums.values())
    mte_total = sum(v for k, v in nums.items() if k.startswith("MTE_"))
    mma_total = sum(v for k, v in nums.items() if k.startswith("MMA_"))
    return {
        "total": total,
        "mte_cycles": mte_total,
        "mma_cycles": mma_total,
        "mte_cycles_pct": pct(mte_total, total),
        "mma_cycles_pct": pct(mma_total, total),
        "shares_pct": {name: pct(value, total) for name, value in nums.items()},
    }

def derive_roofline_metrics(roof: dict[str, Any]) -> dict[str, float | str]:
    base = parse_number(roof.get("MAX_Flops_base"))
    core_clk = parse_number(roof.get("core_clk"))
    case_flops = parse_number(roof.get("case_flops"))
    hbm_bandwidth = parse_number(roof.get("case_bandwith"))
    hbm_peak = parse_number(roof.get("MAX_Bandwith"))
    vl1_i = parse_number(roof.get("case_VL1_I"))
    l2c_i = parse_number(roof.get("case_L2C_I"))
    vl1_peak_gbs = C500_AP_COUNT * C500_VL1_BYTES_PER_CYCLE_PER_AP * core_clk if core_clk else 0.0
    l2c_peak_gbs = C500_L2C_BYTES_PER_CYCLE * core_clk if core_clk else 0.0
    return {
        "peak_trans_tflops": base * 16 if base else NA,
        "peak_fma_tflops": base * 128 if base else NA,
        "peak_mma_fp16_tflops": base * 2048 if base else NA,
        "peak_int8_tflops": base * 4096 if base else NA,
        "hbm_usage_pct": pct(hbm_bandwidth, hbm_peak) if hbm_bandwidth and hbm_peak else NA,
        "vl1_peak_bandwidth_gbs": vl1_peak_gbs if vl1_peak_gbs else NA,
        "l2c_peak_bandwidth_gbs": l2c_peak_gbs if l2c_peak_gbs else NA,
        "vl1_usage_pct": pct(case_flops * 1000.0, vl1_i * vl1_peak_gbs) if vl1_i and vl1_peak_gbs else NA,
        "l2c_usage_pct": pct(case_flops * 1000.0, l2c_i * l2c_peak_gbs) if l2c_i and l2c_peak_gbs else NA,
        "derivation_note": "Roofline peak lines use mcProfiler MAX_Flops_base. HBM usage uses MAX_Bandwith. VL1/L2C usage uses C500 architecture bandwidth constants and core_clk to match the mcProfiler RoofLine chart context.",
    }
