#!/bin/bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO="$ROOT/test/build/demo"
OUT_DIR="$ROOT/test/scoring_baseline"
PHASE="${1:-all}"
case "$PHASE" in
    all|reshape|decode|prefill|tail) ;;
    *) echo "Usage: $0 [all|reshape|decode|prefill|tail]" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"

source /opt/tecoai/setvars.sh
TORCH_LIB="$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null | tail -n 1)"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$DEMO" ]]; then
    cd "$ROOT/test"
    source env.sh
    bash build.sh --arch teco
fi

run_case() {
    local mode="$1"
    local case_path="$2"
    local name
    local rc=0
    name="$(basename "$case_path" .prototxt)"
    echo "===== $name ($mode) ====="
    if [[ "$mode" == "correctness" ]]; then
        (cd "$(dirname "$DEMO")" && ./demo --gid=0 --perf_repeat=5 --warm_repeat=1 \
            --gtest_repeat=1 --case_path="$case_path" 2>&1 |
            tee "$OUT_DIR/$name.log") || rc=$?
    else
        (cd "$(dirname "$DEMO")" && DNN_ONLY_TEST_PERFORMANCE=1 ./demo --gid=0 \
            --perf_repeat=1 --warm_repeat=0 --gtest_repeat=1 \
            --case_path="$case_path" 2>&1 |
            tee "$OUT_DIR/$name.log") || rc=$?
    fi

    # tecotest may return zero even when its log reports failed cases.
    if [[ "$rc" -ne 0 ]] || grep -q "CASES FAILED" "$OUT_DIR/$name.log"; then
        echo "FAILED: $name" >&2
        return 1
    fi
}

if [[ "$PHASE" == "all" || "$PHASE" == "reshape" ]]; then
    for case_path in "$ROOT"/test/zoo/teco/reshape_and_cache/test_case/scoring_*.prototxt; do
        run_case correctness "$case_path"
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "decode" ]]; then
    for case_path in "$ROOT"/test/zoo/teco/flash_attention/test_case/scoring_decode_*.prototxt; do
        run_case correctness "$case_path"
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "prefill" ]]; then
    for case_path in "$ROOT"/test/zoo/teco/flash_attention/test_case/scoring_prefill_*.prototxt; do
        run_case performance "$case_path"
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "tail" ]]; then
    for case_path in "$ROOT"/test/zoo/teco/flash_attention/test_case/scoring_tail_*.prototxt; do
        run_case performance "$case_path"
    done
fi

echo "Baseline logs: $OUT_DIR"
