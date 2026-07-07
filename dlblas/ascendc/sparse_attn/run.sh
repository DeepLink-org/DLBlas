#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# run.sh - Build, generate data, run, and verify sparse_attn
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="sparse_attn"
CASE_ID="${1:-1}"

SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} does not exist"
    echo "=== [2/4] Skip build (reuse existing) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake configure failed"
    make -j4 || die "make build failed"
    cd ..
fi

echo "=== [3/4] Generate test data ==="
cd build
python3 ../scripts/gen_data.py ${CASE_ID} || die "gen_data.py failed"

echo "=== [4/4] Run kernel ==="
rm -f output/output.bin
"./${OP_NAME}" ${CASE_ID} || die "Kernel execution failed (exit code $?)"
[ -f output/output.bin ] || die "output.bin not created (silent failure)"

echo "=== Verify ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "Precision verification FAILED"

echo "=== All passed ==="
exit 0
