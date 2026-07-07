# --------------------------------------------------------------------------
# engram_gate_fwd Ascend C Kernel direct invoke run script
# Build -> Generate data -> Run -> Verify
#
# Usage:
#   bash run.sh              # Full flow (compile + run)
#   bash run.sh --skip-build # Skip compilation
#   bash run.sh --torch      # PyTorch pathway only
# --------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="engram_gate_fwd"

SKIP_BUILD=0
TORCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Set CANN environment ==="
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
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [2/4] Skip build (reuse) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake failed"
    make -j4  || die "make failed"
    cd ..
fi

echo "=== [3/4] Generate test data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/4] Run kernel ==="
rm -f output/output.bin output/output_raw_dot.bin output/output_gate_score.bin \
      output/output_rstd_x.bin output/output_rstd_k.bin
"./${OP_NAME}" || die "Kernel run failed (exit code $?)"
[ -f output/output.bin ] || die "Kernel output.bin not found (silent failure)"
[ -f output/output_raw_dot.bin ] || die "Kernel output_raw_dot.bin not found"
[ -f output/output_gate_score.bin ] || die "Kernel output_gate_score.bin not found"
[ -f output/output_rstd_x.bin ] || die "Kernel output_rstd_x.bin not found"
[ -f output/output_rstd_k.bin ] || die "Kernel output_rstd_k.bin not found"

echo "=== Verification ==="
python3 ../scripts/verify_result.py || die "Verification failed"

echo "=== PyTorch pathway verification (optional) ==="
python3 ../scripts/test_torch.py 2>&1 || echo "NOTE: PyTorch test failed (check separately)"

echo "=== Done ==="
exit 0
