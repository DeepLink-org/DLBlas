#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MACA_PATH="${MACA_PATH:-/opt/maca}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
AUTO_BENCH="${1:-${AUTO_BENCH:-}}"
V0_FILE="${2:-${V0_FILE:-}}"

[[ -n "$AUTO_BENCH" && -f "$AUTO_BENCH" ]] || {
  echo "provide the public evaluator as argument 1 or AUTO_BENCH" >&2
  exit 2
}
[[ -n "$V0_FILE" && -f "$V0_FILE" ]] || {
  echo "provide the Task03 reference as argument 2 or V0_FILE" >&2
  exit 2
}
AUTO_BENCH="$(cd -- "$(dirname -- "$AUTO_BENCH")" && pwd)/$(basename -- "$AUTO_BENCH")"
V0_FILE="$(cd -- "$(dirname -- "$V0_FILE")" && pwd)/$(basename -- "$V0_FILE")"

bash "$ROOT/build.sh"
export LD_LIBRARY_PATH="$ROOT:$MACA_PATH/lib:$MACA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PYTHON_BIN" -B "$AUTO_BENCH" \
  --v0_file "$V0_FILE" --v1_file "$ROOT/task03_v1.py" \
  --warmup 200 --repeat 500
