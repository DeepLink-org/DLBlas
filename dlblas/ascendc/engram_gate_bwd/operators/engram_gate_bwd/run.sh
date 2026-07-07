# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Run script for engram_gate_bwd
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="engram_gate_bwd"
SKIP_BUILD=0
TORCH_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/5] Setting up CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch pathway ==="
    cd build
    python3 ../scripts/test_torch.py || die "PyTorch test failed"
    echo "=== Done ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build: build/${OP_NAME} not found"
    echo "=== [2/5] Skipped build ==="
else
    echo "=== [2/5] Building ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/5] Generating test data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/5] Running kernel ==="
rm -f output/output_*.bin
"./${OP_NAME}" 14 4 128 1e-6 1e-20 || die "Kernel execution failed"

echo "=== [5/5] Verifying results ==="
python3 ../scripts/verify_result.py || die "Verification failed"

echo "=== Done ==="
exit 0
