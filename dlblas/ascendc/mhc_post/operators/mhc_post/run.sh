# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="mhc_post"

SKIP_BUILD=0
TEST_ALL=0
TEST_TAG=""
TORCH_TEST=0

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --test-all)   TEST_ALL=1 ;;
        --test=*)     TEST_TAG="${arg#*=}" ;;
        --torch)      TORCH_TEST=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setting CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH is not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

echo "=== [2/4] Building ==="
if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} does not exist"
    echo "Skipping build (existing binary)"
else
    rm -rf build && mkdir -p build
    cd build
    cmake .. || die "cmake failed"
    make -j4  || die "make failed"
    cd "${SCRIPT_DIR}"
fi

# All test paths run from build/
cd build

run_one_test() {
    local tag="$1" n0="$2" n1="$3" h="$4" mhc_mult="$5"
    echo ""
    echo "--- [${tag}] n0=${n0}, n1=${n1}, h=${h}, mhc=${mhc_mult} ---"

    python3 ../scripts/gen_data.py --test "${tag}" || die "gen_data.py failed for ${tag}"

    echo "--- Running Kernel ---"
    rm -f output/output.bin
    "./${OP_NAME}" "${n0}" "${n1}" "${h}" || die "Kernel failed for ${tag} (exit code $?)"
    [ -f output/output.bin ] || die "output.bin not found after kernel run for ${tag}"

    echo "--- Accuracy verification ---"
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
        "${n0}" "${n1}" "${h}" "${mhc_mult}" \
        || die "Verification failed for ${tag}"
    echo "--- [${tag}] PASSED ---"
}

echo "=== [3/4] Running tests ==="

if [ "${TEST_ALL}" -eq 1 ]; then
    echo "Running ALL test cases..."
    TOTAL=0; PASSED=0; FAILED=0
    while IFS= read -r line; do
        tag=$(echo "$line" | cut -d: -f1)
        case "$tag" in TC-*) ;; *) continue ;; esac
        TOTAL=$((TOTAL + 1))
        n0=$(echo "$line"  | sed -n 's/.*n0=\([0-9]*\).*/\1/p')
        n1=$(echo "$line"  | sed -n 's/.*n1=\([0-9]*\).*/\1/p')
        h=$(echo "$line"   | sed -n 's/.*h=\([0-9]*\).*/\1/p')
        mhc_mult=$(echo "$line" | sed -n 's/.*mhc=\([0-9]*\).*/\1/p')
        if run_one_test "${tag}" "${n0}" "${n1}" "${h}" "${mhc_mult}"; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done < <(python3 ../scripts/gen_data.py --list)

    echo ""
    echo "=========================================="
    echo "ALL TESTS COMPLETE: ${PASSED}/${TOTAL} passed, ${FAILED} failed"
    echo "=========================================="
    [ "${FAILED}" -eq 0 ] || die "Some tests failed"

elif [ -n "${TEST_TAG}" ]; then
    echo "Running single test case: ${TEST_TAG}"
    params=$(python3 ../scripts/gen_data.py --list | grep "^${TEST_TAG}:")
    if [ -z "${params}" ]; then
        die "Unknown test case: ${TEST_TAG}"
    fi
    n0=$(echo "${params}"       | sed -n 's/.*n0=\([0-9]*\).*/\1/p')
    n1=$(echo "${params}"       | sed -n 's/.*n1=\([0-9]*\).*/\1/p')
    h=$(echo "${params}"        | sed -n 's/.*h=\([0-9]*\).*/\1/p')
    mhc_mult=$(echo "${params}" | sed -n 's/.*mhc=\([0-9]*\).*/\1/p')
    run_one_test "${TEST_TAG}" "${n0}" "${n1}" "${h}" "${mhc_mult}"

else
    echo "Running default test case (TC-01)..."
    python3 ../scripts/gen_data.py || die "gen_data.py failed"

    echo "--- Running Kernel ---"
    rm -f output/output.bin
    "./${OP_NAME}" || die "Kernel failed (exit code $?)"
    [ -f output/output.bin ] || die "output.bin not found after kernel run"

    echo "--- Accuracy verification ---"
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
        || die "Verification failed"
fi

echo "=== Done ==="
exit 0
