#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  collect_trace_profile.sh --run-dir DIR --tag TAG --exec-cmd CMD --device ID --target-peu ID --dpg-page-num NUM --target-ap ID --dpc-id IDS --bound-mode MODE --mctracer-bin PATH --cycle-trace-bin PATH --mcprofiler-bin PATH [options]

Collect C500 profiling artifacts with mcTracer, cycle-trace-ng, and mcProfiler,
then run trace_profile_pipeline.py to generate JSON/Markdown reports.
The full collection is retried up to 3 times when required outputs are missing
or invalid.

Required:
  --run-dir DIR          Output run directory, for example profile-artifacts/gemm_v0_baseline
  --tag TAG              Analysis tag, for example baseline or v0_baseline
  --exec-cmd CMD         Profiled command executed from --workdir/current directory
  --device ID            MACA_VISIBLE_DEVICES value
  --target-peu ID        cycle-trace-ng --target-peu value; 1-15, default 1 (0x1) captures PEU 0
  --dpg-page-num NUM     cycle-trace-ng --dpg-page-num value; tool default is 32 (64M)
  --target-ap ID         cycle-trace-ng --target-ap value; tool default is 0
  --dpc-id IDS           cycle-trace-ng --dpc-id value; default 0 captures DPC 0
  --bound-mode MODE      coarse or detailed
  --mctracer-bin PATH    mcTracer path
  --cycle-trace-bin PATH cycle-trace-ng path
  --mcprofiler-bin PATH  mcProfiler path

Options:
  --workdir DIR          Directory to run --exec-cmd from (default: current directory)
  --case-name NAME       mcProfiler --casename value (default: basename of RUN_DIR)
  -h, --help             Show this help

Example:
  .trace-report/scripts/collect_trace_profile.sh \
    --run-dir profile-artifacts/bf16_gemm_kernel_v0_baseline \
    --tag baseline \
    --exec-cmd './test_maca' \
    --device 0 \
    --target-peu 1 \
    --dpg-page-num 32 \
    --target-ap 0 \
    --dpc-id 0 \
    --bound-mode coarse \
    --mctracer-bin /opt/maca/bin/mcTracer \
    --cycle-trace-bin /opt/maca-20250727/bin/cycle-trace-ng \
    --mcprofiler-bin /opt/mcProfiler-linux/mcProfiler
USAGE
}

die() {
  echo "[collect_trace_profile] error: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "[collect_trace_profile] missing/invalid required output: $label ($path)" >&2
    return 1
  fi
}

parse_exec_cmd() {
  local cmd="$1"
  python3 - "$cmd" <<'PY'
import shlex
import sys

for part in shlex.split(sys.argv[1]):
    print(part)
PY
}

require_arg() {
  local option="$1"
  if [[ $# -lt 2 ]]; then
    die "$option requires an argument"
  fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workdir=""
run_dir=""
tag=""
exec_cmd=""
exec_argv=()
case_name=""
device=""
target_peu=""
dpg_page_num=""
target_ap=""
dpc_id=""
bound_mode=""
mctracer_bin=""
cycle_trace_bin=""
mcprofiler_bin=""
profiler_cmdline=""
profiler_out=""
current_marker=""
cycle_trace_extra_args=()

cleanup_success_workdir_outputs() {
  find "$workdir" -maxdepth 1 -type d -name 'tracer_out_*' -exec rm -rf {} +
  find "$workdir" -maxdepth 1 -type f \( -name '.2*.db' -o -name 'c-trace*.json' \) -delete
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) require_arg "$1" "$@"; workdir="$2"; shift 2 ;;
    --run-dir) require_arg "$1" "$@"; run_dir="$2"; shift 2 ;;
    --tag) require_arg "$1" "$@"; tag="$2"; shift 2 ;;
    --exec-cmd) require_arg "$1" "$@"; exec_cmd="$2"; shift 2 ;;
    --case-name) require_arg "$1" "$@"; case_name="$2"; shift 2 ;;
    --device) require_arg "$1" "$@"; device="$2"; shift 2 ;;
    --target-peu) require_arg "$1" "$@"; target_peu="$2"; shift 2 ;;
    --dpg-page-num) require_arg "$1" "$@"; dpg_page_num="$2"; shift 2 ;;
    --target-ap) require_arg "$1" "$@"; target_ap="$2"; shift 2 ;;
    --dpc-id) require_arg "$1" "$@"; dpc_id="$2"; shift 2 ;;
    --bound-mode) require_arg "$1" "$@"; bound_mode="$2"; shift 2 ;;
    --mctracer-bin) require_arg "$1" "$@"; mctracer_bin="$2"; shift 2 ;;
    --cycle-trace-bin) require_arg "$1" "$@"; cycle_trace_bin="$2"; shift 2 ;;
    --mcprofiler-bin) require_arg "$1" "$@"; mcprofiler_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$run_dir" ]] || die "--run-dir is required"
[[ -n "$tag" ]] || die "--tag is required"
[[ -n "$exec_cmd" ]] || die "--exec-cmd is required"
[[ -n "$device" ]] || die "--device is required"
[[ -n "$target_peu" ]] || die "--target-peu is required"
[[ "$target_peu" =~ ^[0-9]+$ ]] || die "--target-peu must be an integer from 1 to 15"
(( target_peu >= 1 && target_peu <= 15 )) || die "--target-peu must be in [1, 15]; 1 (0x1) captures PEU 0"
[[ -n "$dpg_page_num" ]] || die "--dpg-page-num is required"
[[ "$dpg_page_num" =~ ^[0-9]+$ ]] || die "--dpg-page-num must be a positive integer"
(( dpg_page_num > 0 )) || die "--dpg-page-num must be greater than 0"
[[ -n "$target_ap" ]] || die "--target-ap is required"
[[ "$target_ap" =~ ^[0-9]+$ ]] || die "--target-ap must be an integer from 0 to 15"
(( target_ap >= 0 && target_ap <= 15 )) || die "--target-ap must be in [0, 15]"
[[ -n "$dpc_id" ]] || die "--dpc-id is required"
[[ "$dpc_id" =~ ^[0-7](,[0-7])*$ ]] || die "--dpc-id must be a comma-separated list of integers from 0 to 7 with no spaces"
IFS=',' read -ra dpc_id_parts <<< "$dpc_id"
[[ -n "$bound_mode" ]] || die "--bound-mode is required"
[[ "$bound_mode" == "coarse" || "$bound_mode" == "detailed" ]] || die "--bound-mode must be coarse or detailed"
[[ -n "$mctracer_bin" ]] || die "--mctracer-bin is required"
[[ -n "$cycle_trace_bin" ]] || die "--cycle-trace-bin is required"
[[ -n "$mcprofiler_bin" ]] || die "--mcprofiler-bin is required"
mapfile -t exec_argv < <(parse_exec_cmd "$exec_cmd") || die "failed to parse --exec-cmd"
[[ "${#exec_argv[@]}" -gt 0 ]] || die "--exec-cmd did not produce an executable"

workdir="${workdir:-$PWD}"
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
case_name="${case_name:-$(basename "$run_dir")}"
workdir="$(cd "$workdir" && pwd)"
profiler_out="$run_dir/mcprofiler_raw_${tag}"
profiler_cmdline="export MACA_PATH=/opt/maca/; export LD_LIBRARY_PATH=/opt/maca/lib/:\${LD_LIBRARY_PATH}; ${exec_cmd}"

[[ -x "$mctracer_bin" ]] || die "mcTracer not executable: $mctracer_bin"
[[ -x "$cycle_trace_bin" ]] || die "cycle-trace-ng not executable: $cycle_trace_bin"
[[ -x "$mcprofiler_bin" ]] || die "mcProfiler not executable: $mcprofiler_bin"
[[ -f "$script_dir/trace_profile_pipeline.py" ]] || die "trace_profile_pipeline.py not found next to this script"

primary_dpc_id="${dpc_id%%,*}"
cycle_trace_name="c-trace_output_dpc_${primary_dpc_id}.json"
cycle_trace_extra_args=(--dpg-page-num "$dpg_page_num" --target-ap "$target_ap" --dpc-id "$dpc_id")

cleanup_failed_outputs() {
  local marker_path="${1:-}"
  rm -f \
    "$run_dir/tracer_out.json" \
    "$run_dir"/c-trace_output_dpc_*.json \
    "$run_dir/mcprofiler_report_dumped.json" \
    "$run_dir/mcprofiler_report.txt.json" \
    "$run_dir/REPORT_${tag}.md" \
    "$run_dir/analysis/digest_${tag}.md" \
    "$run_dir/analysis/metrics_all_${tag}.json" \
    "$run_dir/analysis/metrics_key_${tag}.json"
  rm -rf "$run_dir/artifacts" "$profiler_out"
  if [[ -n "$marker_path" && -f "$marker_path" ]]; then
    find "$run_dir" -maxdepth 1 -type f -name '*.png*' -newer "$marker_path" -delete
    rm -f "$marker_path"
  fi
}

collect_once() {
  local attempt="$1"
  local marker
  cleanup_failed_outputs
  marker="$(mktemp "$run_dir/.collect_attempt_${attempt}.XXXXXX")"
  current_marker="$marker"

  echo "[Attempt $attempt/3] Starting trace profile collection"

  echo "[Step 1/4] Running mcTracer..."
  MACA_VISIBLE_DEVICES="$device" "$mctracer_bin" "${exec_argv[@]}" || return 1
  tracer_json="$(find . -path './tracer_out_*/tracer_out-*.json' -newer "$marker" -type f 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "$tracer_json" ]] || {
    echo "[Step 1/4] mcTracer output not found: tracer_out_*/tracer_out-*.json" >&2
    return 1
  }
  cp "$tracer_json" "$run_dir/tracer_out.json" || return 1
  require_file "$run_dir/tracer_out.json" "mcTracer JSON" || return 1
  echo "[Step 1/4] mcTracer -> $run_dir/tracer_out.json"

  echo "[Step 2/4] Running cycle-trace-ng..."
  ENABLE_DPG=1 ENABLE_DPG_DUMP=1 ISU_FASTMODEL=0 MACA_VISIBLE_DEVICES="$device" \
    "$cycle_trace_bin" "${exec_argv[@]}" --format json --target-peu "$target_peu" "${cycle_trace_extra_args[@]}" || return 1
  cycle_json="$(find . -maxdepth 1 -name "$cycle_trace_name" -newer "$marker" -type f 2>/dev/null | sort | head -n 1 || true)"
  [[ -n "$cycle_json" ]] || {
    echo "[Step 2/4] CycleTrace output not found: $cycle_trace_name" >&2
    return 1
  }
  for part in "${dpc_id_parts[@]}"; do
    local dpc_trace_name="c-trace_output_dpc_${part}.json"
    local dpc_cycle_json
    dpc_cycle_json="$(find . -maxdepth 1 -name "$dpc_trace_name" -newer "$marker" -type f 2>/dev/null | sort | head -n 1 || true)"
    if [[ -n "$dpc_cycle_json" ]]; then
      cp "$dpc_cycle_json" "$run_dir/$dpc_trace_name" || return 1
      echo "[Step 2/4] CycleTrace -> $run_dir/$dpc_trace_name"
    fi
  done
  require_file "$run_dir/$cycle_trace_name" "CycleTrace JSON" || return 1

  echo "[Step 3/4] Running mcProfiler..."
  rm -rf "$profiler_out"
  MACA_VISIBLE_DEVICES="$device" "$mcprofiler_bin" perf_exec \
    --cmdline "$profiler_cmdline" \
    --casename "$case_name" \
    --output "$profiler_out" || return 1

  [[ -d "$profiler_out" ]] || {
    echo "[Step 3/4] mcProfiler output directory not found: $profiler_out" >&2
    return 1
  }
  if [[ -f "$profiler_out/report_dumped_result.json" ]]; then
    cp "$profiler_out/report_dumped_result.json" "$run_dir/mcprofiler_report_dumped.json" || return 1
  elif [[ -f "$profiler_out/mcprofiler_report_dumped.json" ]]; then
    cp "$profiler_out/mcprofiler_report_dumped.json" "$run_dir/mcprofiler_report_dumped.json" || return 1
  fi
  require_file "$run_dir/mcprofiler_report_dumped.json" "mcProfiler dumped JSON" || return 1

  if [[ -f "$profiler_out/report.txt.json" ]]; then
    cp "$profiler_out/report.txt.json" "$run_dir/mcprofiler_report.txt.json" || return 1
  elif [[ -f "$profiler_out/mcprofiler_report.txt.json" ]]; then
    cp "$profiler_out/mcprofiler_report.txt.json" "$run_dir/mcprofiler_report.txt.json" || return 1
  fi
  require_file "$run_dir/mcprofiler_report.txt.json" "mcProfiler report text JSON" || return 1

  find "$profiler_out" -maxdepth 1 -name '*.png*' -type f -exec cp {} "$run_dir/" \;
  rm -rf "$profiler_out"
  echo "[Step 3/4] mcProfiler artifacts normalized under $run_dir"

  echo "[Step 4/4] Running trace_profile_pipeline.py..."
  python3 "$script_dir/trace_profile_pipeline.py" run \
    --source "$run_dir" \
    --run-dir "$run_dir" \
    --tag "$tag" \
    --bound-mode "$bound_mode" \
    --cycle-dpc-id "$dpc_id" || return 1

  require_file "$run_dir/artifacts/tracer_out.json" "archived mcTracer JSON" || return 1
  require_file "$run_dir/artifacts/$cycle_trace_name" "archived CycleTrace JSON" || return 1
  require_file "$run_dir/artifacts/mcprofiler_report_dumped.json" "archived mcProfiler dumped JSON" || return 1
  require_file "$run_dir/artifacts/mcprofiler_report.txt.json" "archived mcProfiler report text JSON" || return 1
  require_file "$run_dir/analysis/digest_${tag}.md" "digest Markdown" || return 1
  require_file "$run_dir/analysis/metrics_all_${tag}.json" "full metrics JSON" || return 1
  require_file "$run_dir/analysis/metrics_key_${tag}.json" "key metrics JSON" || return 1
  require_file "$run_dir/REPORT_${tag}.md" "final report" || return 1

  rm -f "$marker"
  current_marker=""
}

echo "=========================================="
echo " Trace profile collection"
echo " workdir:      $workdir"
echo " run_dir:      $run_dir"
echo " tag:          $tag"
echo " exec_cmd:     $exec_cmd"
echo " case_name:    $case_name"
echo " device:       $device"
echo " target_peu:   $target_peu"
echo " dpg_page_num: $dpg_page_num"
echo " target_ap:    $target_ap"
echo " dpc_id:       $dpc_id"
echo " bound_mode:   $bound_mode"
echo "=========================================="

cd "$workdir"

success=0
for attempt in 1 2 3; do
  if collect_once "$attempt"; then
    success=1
    break
  fi
  if [[ "$attempt" -lt 3 ]]; then
    echo "[Attempt $attempt/3] Required outputs missing/invalid or command failed; deleting failed outputs before retry." >&2
  else
    echo "[Attempt $attempt/3] Required outputs missing/invalid or command failed; deleting failed outputs; no retries left." >&2
  fi
  cleanup_failed_outputs "$current_marker"
  current_marker=""
done

if [[ "$success" -ne 1 ]]; then
  die "trace profile collection failed after 3 attempts. Failed outputs were deleted; report this to the user and wait for a decision before changing parameters or continuing."
fi

cleanup_success_workdir_outputs

echo "=========================================="
echo " Trace profile complete"
echo " report: $run_dir/REPORT_${tag}.md"
echo " metrics: $run_dir/analysis/metrics_all_${tag}.json"
echo "=========================================="
