"""Small numeric, path, and formatting helpers for trace reports."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .constants import NA

TENSOR_PATH_KERNEL_TOKENS = (
    "gemm",
    "matmul",
    "mma",
    "attention",
    "conv",
)

def pct(part: float, total: float) -> float:
    return round(part * 100.0 / total, 4) if total else 0.0

def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))

def sliding_window_peak(values: Sequence[int], width: int) -> int:
    best = 0
    left = 0
    for right, ts in enumerate(values):
        while ts - values[left] > width:
            left += 1
        best = max(best, right - left + 1)
    return best

def parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if "(" in cleaned:
            cleaned = cleaned.split("(", 1)[0].strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def parse_optional_number(value: Any) -> float | str:
    if value is None:
        return NA
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and not value.strip():
        return NA
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if "(" in cleaned:
            cleaned = cleaned.split("(", 1)[0].strip()
        try:
            return float(cleaned)
        except ValueError:
            return NA
    return NA

def as_number(value: Any, default: float = 0.0) -> float:
    if value == NA:
        return default
    return parse_number(value)

def is_available_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if value == NA:
        return NA
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)

def fmt_pct(value: Any) -> str:
    if value == NA or value is None:
        return NA
    return f"{as_number(value):.2f}%"

def fmt_float(value: Any, digits: int = 2) -> str:
    if value == NA or value is None:
        return NA
    return f"{as_number(value):.{digits}f}"

def ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0

def numeric_values(values: dict[str, Any]) -> list[float]:
    return [parse_number(v) for v in values.values()]

def expects_tensor_path(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
    return any(token in lowered for token in TENSOR_PATH_KERNEL_TOKENS)

def has_shared_memory(launch: dict[str, Any]) -> bool:
    return (
        as_number(launch.get("static_shared_bytes")) > 0
        or as_number(launch.get("dynamic_shared_bytes")) > 0
        or as_number(launch.get("shared_memory_occupancy_pct")) > 0
    )

def has_wsm_activity(
    launch: dict[str, Any],
    cycle: dict[str, Any],
    profiler: dict[str, Any],
) -> bool:
    return (
        has_shared_memory(launch)
        or as_number(cycle.get("bsm", 0)) > 0
        or as_number(cycle.get("bsm_pct", 0)) > 0
        or as_number(profiler.get("shared_memory_load_instructions", 0)) > 0
        or as_number(profiler.get("shared_memory_store_instructions", 0)) > 0
    )

def has_bank_conflict_signal(
    launch: dict[str, Any],
    cycle: dict[str, Any],
    profiler: dict[str, Any],
) -> bool:
    conflict_cycles = as_number(profiler.get("avg_conflict_cycles_per_inst"))
    efficiency = as_number(profiler.get("shared_memory_efficiency_pct"))
    return conflict_cycles > 0.5 or (
        has_wsm_activity(launch, cycle, profiler)
        and efficiency > 0
        and efficiency < 80
    )

def min_avg_max(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), sum(values) / len(values), max(values)
