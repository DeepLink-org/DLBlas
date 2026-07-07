#!/bin/bash
# ============================================================================
# Big Fuse MHC Pre-processing Fused Kernel - Build & Run Script
#
# Usage:
#   bash run.sh              # Full flow (build + test + verify)
#   bash run.sh --skip-build # Skip build, reuse existing artifacts
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="big_fuse"
SKIP_BUILD=0

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/5] Setting up CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} not found"
    echo "=== [2/5] Skipping build (reusing artifacts) ==="
else
    echo "=== [2/5] Building ==="
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/5] Generating test data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/5] Computing golden (PyTorch reference) ==="
python3 ../scripts/golden.py || die "golden.py failed"

echo "=== [5/5] Running kernels ==="
rm -f output/post_mix.bin output/comb_mix.bin output/layer_input.bin
"./${OP_NAME}" || die "Kernel execution failed (exit code $?)"

# Verify all three outputs exist
for f in output/post_mix.bin output/comb_mix.bin output/layer_input.bin; do
    [ -f "$f" ] || die "Output file $f not found (silent failure)"
done

echo ""
echo "=== Precision Verification ==="
python3 ../scripts/verify_result.py
VERIFY_EXIT=$?
cd ..

echo ""
if [ $VERIFY_EXIT -eq 0 ]; then
    echo "=== SUCCESS: All tests passed! ==="
else
    echo "=== FAILURE: Some tests failed (exit code $VERIFY_EXIT) ==="
    exit $VERIFY_EXIT
fi
