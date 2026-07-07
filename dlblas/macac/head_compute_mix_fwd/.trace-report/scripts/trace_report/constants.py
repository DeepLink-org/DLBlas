"""Shared constants for trace-report pipeline."""

from __future__ import annotations

import re


def validate_dpc_id(dpc_id: str | int | None = None) -> str:
    value = "0" if dpc_id is None else str(dpc_id).strip()
    if not value:
        value = "0"
    if not re.fullmatch(r"[0-7](,[0-7])*", value):
        raise ValueError("DPC id must be a comma-separated list of integers from 0 to 7 with no spaces")
    return value


def primary_dpc_id(dpc_id: str | int | None = None) -> str:
    return validate_dpc_id(dpc_id).split(",", 1)[0]


def dpc_ids(dpc_id: str | int | None = None) -> tuple[str, ...]:
    return tuple(validate_dpc_id(dpc_id).split(","))


def cycle_trace_filename(dpc_id: str | int | None = None) -> str:
    return f"c-trace_output_dpc_{primary_dpc_id(dpc_id)}.json"


def cycle_trace_filenames(dpc_id: str | int | None = None) -> tuple[str, ...]:
    return tuple(f"c-trace_output_dpc_{part}.json" for part in dpc_ids(dpc_id))


def raw_filenames(dpc_id: str | int | None = None) -> tuple[str, ...]:
    return (
        *cycle_trace_filenames(dpc_id),
        "tracer_out.json",
        "mcprofiler_report_dumped.json",
        "mcprofiler_report.txt.json",
    )


RAW_FILENAMES = raw_filenames()

NA = "N/A"

KEY_METRICS = (
    ("kernel.span_cycles", "Kernel span", "cycles", "lower"),
    ("cycle.total_instructions", "CycleTrace instructions", "inst", "context"),
    ("cycle.mma_pct", "CycleTrace MMA share", "%", "higher"),
    ("cycle.mte_pct", "CycleTrace MTE share", "%", "context"),
    ("cycle.bsm_pct", "CycleTrace BSM share", "%", "context"),
    ("cycle.gvm_pct", "CycleTrace GVM share", "%", "lower"),
    ("cycle.arrive_pct", "CycleTrace ARRIVE share", "%", "lower"),
    ("cycle.ldu_pct", "CycleTrace LDU share", "%", "context"),
    ("cycle.nop_pct", "CycleTrace NOP share", "%", "lower"),
    ("cycle.wif_mean", "Wave in flight mean", "waves", "higher"),
    ("cycle.wif_max", "Wave in flight max", "waves", "higher"),
    ("cycle.wif_p25", "Wave in flight P25", "waves", "context"),
    ("cycle.wif_p50", "Wave in flight P50", "waves", "context"),
    ("cycle.wif_p75", "Wave in flight P75", "waves", "context"),
    ("cycle.gvm_600cy_peak", "GVM issue peak / 600 cycles", "inst", "lower"),
    ("launch.registers_per_thread", "Registers/thread", "regs", "lower"),
    ("launch.static_shared_bytes", "Static shared", "bytes", "lower"),
    ("launch.mtreg_occupancy_pct", "mtreg occupancy", "%", "higher"),
    ("launch.shared_memory_occupancy_pct", "shared memory occupancy", "%", "higher"),
    ("launch.effective_occupancy_pct", "digest occupancy bound", "%", "higher"),
    ("profiler.ap_mte_duty_pct", "AP MTE duty", "%", "higher"),
    ("profiler.ap_ste_duty_pct", "AP STE duty", "%", "higher"),
    ("profiler.ap_mma_duty_pct", "AP MMA duty", "%", "higher"),
    ("profiler.real_ipc", "Real IPC", "inst/cycle", "higher"),
    ("profiler.instruction_throughput_efficiency_pct", "Instruction throughput efficiency", "%", "higher"),
    ("profiler.instructions_per_ap", "Instructions per AP", "inst/AP", "context"),
    ("profiler.compute_instruction_cycles_summary.mma_cycles_pct", "MMA compute-cycle share", "%", "higher"),
    ("profiler.isu_stall_summary.top_pct", "Top ISU stall share", "%", "lower"),
    ("profiler.isu_stall_summary.total", "ISU stall cycles", "cycles", "lower"),
    ("profiler.vl1_hit_rate_pct", "VL1 hit rate", "%", "higher"),
    ("profiler.l2c_hit_rate_pct", "L2C hit rate", "%", "higher"),
    ("profiler.dnoc_read_req", "DNOC read requests", "req", "lower"),
    ("profiler.dnoc_write_req", "DNOC write requests", "req", "lower"),
    ("profiler.global_memory_read_bytes", "Global memory read bytes", "bytes", "lower"),
    ("profiler.global_memory_write_bytes", "Global memory write bytes", "bytes", "lower"),
    ("profiler.shared_memory_efficiency_pct", "Shared memory efficiency", "%", "higher"),
    ("profiler.avg_conflict_cycles_per_inst", "Avg conflict cycles/inst", "cycles", "lower"),
    ("profiler.achieved_waves", "Achieved waves", "waves", "higher"),
    ("profiler.dispatched_waves", "Dispatched waves", "waves", "context"),
    ("profiler.average_wave_life_cycles", "Average wave life", "cycles", "lower"),
    ("profiler.achieved_flops_tflops", "Empirical achieved FLOPS", "TFLOPS", "higher"),
    ("profiler.achieved_bandwidth_gbs", "Empirical achieved bandwidth", "GB/s", "higher"),
    ("profiler.achieved_intensity_flop_per_byte", "Empirical roofline intensity", "FLOP/Byte", "context"),
    ("profiler.roofline.hbm_usage_pct", "Roofline HBM usage", "%", "higher"),
    ("profiler.roofline.vl1_usage_pct", "Roofline VL1 usage", "%", "higher"),
    ("profiler.roofline.l2c_usage_pct", "Roofline L2C usage", "%", "higher"),
    ("profiler.roofline.peak_fma_tflops", "Roofline FMA peak", "TFLOP/s", "context"),
    ("profiler.roofline.peak_mma_fp16_tflops", "Roofline MMA-FP16 peak", "TFLOP/s", "context"),
    ("profiler.roofline.case_VL1_I", "VL1-level roofline intensity", "FLOP/Byte", "context"),
    ("profiler.roofline.case_L2C_I", "L2C-level roofline intensity", "FLOP/Byte", "context"),
    ("profiler.dpc_compute_imbalance_pct", "DPC compute imbalance", "%", "lower"),
)
