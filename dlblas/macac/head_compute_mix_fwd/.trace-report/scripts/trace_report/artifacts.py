"""Artifact collection and JSON file helpers."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .constants import cycle_trace_filename, cycle_trace_filenames, raw_filenames

HARDWARE_CATS = ("MTE", "STE", "MMA", "BSM", "GLOBAL", "ARRIVE", "LDU")

def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def validate_cycle_trace(path: Path) -> list[str]:
    try:
        raw = load_json(path)
    except Exception as exc:
        return [f"CycleTrace JSON is not readable: {exc}"]
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else []
    if not isinstance(events, list) or not events:
        return ["CycleTrace JSON has no traceEvents"]
    hardware_events = [
        event for event in events
        if isinstance(event, dict) and event.get("cat") in HARDWARE_CATS
    ]
    if not hardware_events:
        return ["CycleTrace JSON has no hardware instruction events"]
    return []

def latest_pngs(source: Path) -> list[Path]:
    by_chart: dict[str, Path] = {}
    for png in sorted(source.glob("*.png*")):
        match = re.match(r"^(?P<chart>.*?)(?P<stamp>\d{14,})(?P<suffix>\.png.*)$", png.name)
        key = match.group("chart") if match else png.stem
        previous = by_chart.get(key)
        if previous is None or png.name > previous.name:
            by_chart[key] = png
    return sorted(by_chart.values(), key=lambda path: path.name)

def collect_artifacts(source: Path, run_dir: Path, tag: str, cycle_dpc_id: str | int | None = None) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"source is not a directory: {source}")

    dest = run_dir / "artifacts"
    same_source_and_dest = source.resolve() == dest.resolve()
    if dest.exists() and not same_source_and_dest:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cycle_names = cycle_trace_filenames(cycle_dpc_id)
    raw_names = raw_filenames(cycle_dpc_id)
    copied: list[str] = []
    for name in raw_names:
        src = source / name
        if src.exists():
            if not same_source_and_dest:
                shutil.copy2(src, dest / name)
            copied.append(name)

    source_pngs = list(source.glob("*.png*"))
    for png in latest_pngs(source):
        if not same_source_and_dest:
            shutil.copy2(png, dest / png.name)
        copied.append(png.name)

    manifest = {
        "tag": tag,
        "source": str(source),
        "artifact_dir": str(dest),
        "copied_files": copied,
        "cycle_trace_primary": cycle_trace_filename(cycle_dpc_id),
        "cycle_trace_files": [name for name in cycle_names if name in copied],
        "missing_required": [
            name for name in (*cycle_names, "tracer_out.json")
            if name not in copied
        ],
        "missing_optional": [
            name for name in ("mcprofiler_report_dumped.json", "mcprofiler_report.txt.json")
            if name not in copied
        ],
        "invalid_required": [],
    }
    cycle_path = dest / cycle_trace_filename(cycle_dpc_id)
    if cycle_path.exists():
        manifest["invalid_required"].extend(validate_cycle_trace(cycle_path))
    write_json(dest / "collection_manifest.json", manifest)
    if manifest["missing_required"] or manifest["invalid_required"]:
        raise ValueError(
            "required artifacts are incomplete: "
            f"missing={manifest['missing_required']}, invalid={manifest['invalid_required']}"
        )
    if source.resolve() == run_dir.resolve():
        for name in raw_names:
            path = source / name
            if path.exists():
                path.unlink()
        for png in source_pngs:
            if png.exists():
                png.unlink()
    return dest

def find_artifact_dir(run_dir: Path, tag: str, artifact_dir: Path | None) -> Path:
    if artifact_dir is not None:
        return artifact_dir
    return run_dir / "artifacts"
