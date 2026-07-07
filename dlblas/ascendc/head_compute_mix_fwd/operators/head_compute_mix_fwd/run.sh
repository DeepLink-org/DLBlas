# ============================================================================
# head_compute_mix_fwd run script
#   Compile → Generate test data → Run kernel → Verify precision → PyTorch test
#
# Usage:
#   bash run.sh                # Full pipeline
#   bash run.sh --skip-build   # Skip compilation
#   bash run.sh --torch        # PyTorch pathway only
#   bash run.sh --skip-build --torch  # Skip compilation, PyTorch only
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="head_compute_mix_fwd"

SKIP_BUILD=0
TORCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch pathway verification ==="
    cd build
    python3 ../scripts/gen_data.py 16 16384 4
    python3 ../scripts/test_torch.py || die "PyTorch test failed"
    echo "=== Done ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} not found"
    echo "=== [2/4] Skip compilation (reusing existing) ==="
else
    echo "=== [2/4] Compilation ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generate test data ==="
cd build
python3 ../scripts/gen_data.py "$@" 2>/dev/null || python3 ../scripts/gen_data.py

echo "=== [4/4] Run kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel execution failed (exit code $?)"
[ -f output/output.bin ] || die "output.bin not found after kernel execution"

echo "=== Precision verification ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "Precision verification failed"

echo "=== PyTorch pathway verification ==="
python3 ../scripts/test_torch.py \
    || die "PyTorch test failed"

echo "=== All tests passed! ==="
exit 0
