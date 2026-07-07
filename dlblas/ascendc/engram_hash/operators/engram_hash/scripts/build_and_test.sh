#!/bin/bash
# engram_hash AscendC — Build, Verify (bit-exact), Benchmark runner.
# Usage: bash scripts/build_and_test.sh [--clean] [--full] [--bench]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${OP_DIR}/build"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2}"
export EH_IO_DIR="${OP_DIR}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh

CLEAN=false; FULL=false; BENCH=false
for a in "$@"; do case $a in
    --clean) CLEAN=true ;;
    --full)  FULL=true ;;
    --bench) BENCH=true ;;
esac; done

echo "=== engram_hash build (device ${ASCEND_RT_VISIBLE_DEVICES}) ==="
[ "$CLEAN" = true ] && rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${OP_DIR}" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make engram_hash_custom engram_hash_ops -j"$(nproc)" 2>&1 | tail -6
ls -lh "${BUILD_DIR}/engram_hash_custom" "${BUILD_DIR}/libengram_hash_ops.so"

echo ""
echo "=== gen baseline data + direct-invoke verify ==="
python3 "${SCRIPT_DIR}/gen_data.py" --nt 4096 --ngram 3 --layers 2 --tables 8
"${BUILD_DIR}/engram_hash_custom" 4096 3 2 8 0 48
python3 "${SCRIPT_DIR}/verify_result.py"

echo ""
echo "=== torch integration test ==="
python3 "${SCRIPT_DIR}/test_torch.py"

if [ "$FULL" = true ]; then
    echo ""
    echo "=== full verification matrix ==="
    python3 "${SCRIPT_DIR}/run_verify_matrix.py"
fi

if [ "$BENCH" = true ]; then
    echo ""
    echo "=== benchmark ==="
    python3 "${SCRIPT_DIR}/benchmark.py"
fi

echo ""
echo "=== engram_hash done ==="
