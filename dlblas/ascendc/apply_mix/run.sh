# apply_mix AscendC kernel build and test script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="apply_mix"

SKIP_BUILD=0
TORCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Set CANN Environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch pathway verification ==="
    cd build
    python3 ../scripts/test_torch.py || die "PyTorch test failed"
    echo "=== Done ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build: build/${OP_NAME} not found"
    echo "=== [2/4] Skip build (reusing existing) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake configure failed"
    make -j4  || die "make build failed"
    cd ..
fi

echo "=== [3/4] Generate Test Data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/4] Run Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel run failed (exit code $?)"
[ -f output/output.bin ] || die "output/output.bin missing after kernel run"

# Read shape from shape.bin to pass to verifier
echo "=== Precision Verification ==="
python3 -c "
import numpy as np
shape = np.fromfile('input/shape.bin', dtype=np.uint32)
print(f'n0={shape[0]}, n1={shape[1]}, h={shape[3]}')
" || die "Failed to read shape"

# Run verification
N0=$(python3 -c "import numpy as np; print(np.fromfile('input/shape.bin', dtype=np.uint32)[0])")
N1=$(python3 -c "import numpy as np; print(np.fromfile('input/shape.bin', dtype=np.uint32)[1])")
H=$(python3 -c "import numpy as np; print(np.fromfile('input/shape.bin', dtype=np.uint32)[3])")

python3 ../scripts/verify_result.py output/output.bin output/golden.bin ${N0} ${N1} ${H} \
    || die "Precision verification FAILED"

echo "=== PyTorch Pathway Verification ==="
python3 ../scripts/test_torch.py \
    || die "PyTorch test FAILED"

echo "=== Done ==="
exit 0
