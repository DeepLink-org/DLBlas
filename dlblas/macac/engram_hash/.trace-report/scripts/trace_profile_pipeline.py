#!/usr/bin/env python3
"""CLI wrapper for C500 trace artifact collection, analysis, and reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from trace_report.artifacts import collect_artifacts
from trace_report.compare import compare_cases, compare_tags
from trace_report.metrics import analyze_artifacts
from trace_report.report_append import append_llm_followup_diagnosis
from trace_report.target import TARGET_ENV_MARKER, TARGET_ENV_MARKER_VALUE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C500 trace artifact collect/analyze/report pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect", help="Copy raw trace artifacts into <run-dir>/artifacts/")
    collect.add_argument("--source", type=Path, required=True)
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--tag", required=True)
    collect.add_argument("--cycle-dpc-id", default=os.environ.get("CYCLE_TRACE_DPC_ID", "0"))

    analyze = sub.add_parser("analyze", help="Analyze artifacts and write <run-dir>/analysis/")
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--tag", required=True)
    analyze.add_argument("--artifact-dir", type=Path)
    analyze.add_argument("--cycle-dpc-id", default=os.environ.get("CYCLE_TRACE_DPC_ID", "0"))
    analyze.add_argument(
        "--bound-mode",
        choices=("coarse", "detailed"),
        default="coarse",
        help="Bound classification mode. Default: coarse (compute/memory/latency/occupancy).",
    )

    run = sub.add_parser("run", help="Collect then analyze one profiling artifact directory.")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--cycle-dpc-id", default=os.environ.get("CYCLE_TRACE_DPC_ID", "0"))
    run.add_argument(
        "--bound-mode",
        choices=("coarse", "detailed"),
        default="coarse",
        help="Bound classification mode. Default: coarse (compute/memory/latency/occupancy).",
    )

    compare = sub.add_parser("compare", help="Compare already analyzed tags.")
    compare.add_argument("--run-dir", type=Path, help="Run directory containing analysis/metrics_all_<tag>.json files.")
    compare.add_argument("--tag", action="append", help="Tag in --run-dir. Repeat for same-run-dir comparisons.")
    compare.add_argument("--case", action="append", help="Cross-run comparison case as label=/path/to/run-dir. Repeat at least twice.")
    compare.add_argument("--output-dir", type=Path, help="Output directory for --case comparison. Default: first case analysis dir.")

    append = sub.add_parser("append-diagnosis", help="Append LLM follow-up diagnosis from stdin to target REPORT_<tag>.md.")
    append.add_argument("--report", type=Path, required=True, help="Path to REPORT_<tag>.md.")

    return parser


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "collect":
            dest = collect_artifacts(args.source, args.run_dir, args.tag, args.cycle_dpc_id)
            print(f"[collect] {args.tag}: {dest}")
        elif args.cmd == "analyze":
            analyze_artifacts(args.run_dir, args.tag, args.artifact_dir, args.bound_mode, args.cycle_dpc_id)
            print(f"[analyze] {args.tag}: {args.run_dir / 'analysis'}")
        elif args.cmd == "run":
            dest = collect_artifacts(args.source, args.run_dir, args.tag, args.cycle_dpc_id)
            analyze_artifacts(args.run_dir, args.tag, dest, args.bound_mode, args.cycle_dpc_id)
            print(f"[run] {args.tag}: {args.run_dir}")
        elif args.cmd == "compare":
            if args.case:
                out = compare_cases(args.case, args.output_dir)
            else:
                if args.run_dir is None or not args.tag:
                    raise ValueError("compare requires either --case repeated at least twice, or --run-dir with repeated --tag")
                out = compare_tags(args.run_dir, args.tag)
            print(f"[compare] {out}")
        elif args.cmd == "append-diagnosis":
            if os.environ.get(TARGET_ENV_MARKER) != TARGET_ENV_MARKER_VALUE:
                raise ValueError("append-diagnosis must be run through trace_report_env.py --config \"$ACTIVE_CONFIG\" run so it writes the target report")
            append_llm_followup_diagnosis(args.report, sys.stdin.read())
            print(f"[append-diagnosis] {args.report}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
