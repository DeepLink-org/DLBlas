# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Ascend C Kernel Direct Invoke - head_compute_mix_bwd
# Build -> Generate data -> Run -> Verify
#
# Usage:
#   bash run.sh                # Full flow (build + standard test)
#   bash run.sh --skip-build   # Skip build, reuse existing artifacts
#   bash run.sh --torch        # Only test PyTorch pathway (comprehensive)
#   bash run.sh --all          # Full build + comprehensive test suite
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="head_compute_mix_bwd"

SKIP_BUILD=0
TORCH_ONLY=0
ALL_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
        --all)        ALL_TESTS=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Set CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch pathway comprehensive test ==="
    cd build
    python3 ../scripts/test_torch.py || die "PyTorch test failed"
    echo "=== Done ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} not found"
    echo "=== [2/4] Skip build (reusing artifacts) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake configure failed"
    make -j4 || die "make build failed"
    cd ..
fi

if [ "${ALL_TESTS}" -eq 1 ]; then
    echo "=== [3/4] Comprehensive test suite (direct invoke) ==="
    cd build
    python3 ../scripts/test_all.py || die "Comprehensive test failed"
    echo ""
    echo "=== [4/4] PyTorch pathway comprehensive test ==="
    python3 ../scripts/test_torch.py || die "PyTorch test failed"
else
    echo "=== [3/4] Generate test data ==="
    cd build
    python3 ../scripts/gen_data.py || die "gen_data.py failed"

    echo "=== [4/4] Run Kernel ==="
    rm -f output/output_grad_input_mix.bin
    rm -f output/output_grad_mhc_scale.bin
    rm -f output/output_grad_mhc_base.bin
    "./${OP_NAME}" || die "Kernel execution failed (exit code $?)"

    # Check all output files exist
    [ -f output/output_grad_input_mix.bin ] || die "output_grad_input_mix.bin not found"
    [ -f output/output_grad_mhc_scale.bin ] || die "output_grad_mhc_scale.bin not found"
    [ -f output/output_grad_mhc_base.bin ] || die "output_grad_mhc_base.bin not found"

    echo "=== Precision verification ==="
    python3 ../scripts/verify_result.py \
        || die "Precision verification failed"

    echo "=== PyTorch pathway test ==="
    python3 ../scripts/test_torch.py \
        || die "PyTorch test failed"
fi

echo "=== Done ==="
exit 0
