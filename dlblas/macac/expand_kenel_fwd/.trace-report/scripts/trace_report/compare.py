"""Comparison report generation for analyzed trace tags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .artifacts import load_json
from .constants import KEY_METRICS
from .utils import fmt, get_path

def compare_tags(run_dir: Path, tags: Sequence[str]) -> Path:
    if len(tags) < 2:
        raise ValueError("compare requires at least two --tag values")
    analysis_dir = run_dir / "analysis"
    reports = {
        tag: load_json(analysis_dir / f"metrics_all_{tag}.json")
        for tag in tags
    }
    return write_compare(analysis_dir, tags, reports)

def compare_cases(case_specs: Sequence[str], output_dir: Path | None = None) -> Path:
    if len(case_specs) < 2:
        raise ValueError("compare requires at least two --case values")
    labels: list[str] = []
    reports: dict[str, Any] = {}
    first_analysis_dir: Path | None = None
    for spec in case_specs:
        label, run_dir = parse_case_spec(spec)
        analysis_dir = run_dir / "analysis"
        if first_analysis_dir is None:
            first_analysis_dir = analysis_dir
        metric_path = single_metrics_file(analysis_dir)
        labels.append(label)
        reports[label] = load_json(metric_path)
    out_dir = output_dir or first_analysis_dir
    if out_dir is None:
        raise ValueError("compare output directory could not be determined")
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_compare(out_dir, labels, reports)

def parse_case_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--case must use label=/path/to/run-dir")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("--case label must not be empty")
    return label, Path(raw_path)

def single_metrics_file(analysis_dir: Path) -> Path:
    matches = sorted(analysis_dir.glob("metrics_all_*.json"))
    if not matches:
        raise FileNotFoundError(f"no metrics_all_*.json found in {analysis_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple metrics_all_*.json files found in {analysis_dir}; "
            "use one analyzed tag per run directory for --case comparison"
        )
    return matches[0]

def write_compare(analysis_dir: Path, tags: Sequence[str], reports: dict[str, Any]) -> Path:
    first = reports[tags[0]]
    last = reports[tags[-1]]
    lines = [
        f"# Trace Profile Compare: {' vs '.join(tags)}",
        "",
        "| Metric | Unit | Better | " + " | ".join(tags) + " | Delta |",
        "|---|---|---|" + "---:|" * len(tags) + "---:|",
    ]
    for path, label, unit, better in KEY_METRICS:
        values = [get_path(reports[tag], path) for tag in tags]
        lines.append(
            f"| {label} | {unit} | {better} | "
            + " | ".join(fmt(v) for v in values)
            + f" | {format_delta(get_path(first, path), get_path(last, path))} |"
        )
    out = analysis_dir / f"compare_{'_vs_'.join(tags)}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

def format_delta(before: Any, after: Any) -> str:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return ""
    diff = after - before
    if before:
        return f"{diff:+.4g} ({diff * 100.0 / abs(before):+.2f}%)"
    return f"{diff:+.4g}"
