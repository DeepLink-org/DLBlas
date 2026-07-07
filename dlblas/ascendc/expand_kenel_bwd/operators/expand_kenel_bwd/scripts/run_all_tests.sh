#!/bin/bash
# Run all 7 test cases for expand_kenel_bwd
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd to operator root directory
cd "${SCRIPT_DIR}/.."

OP_NAME="expand_kenel_bwd"

source "${ASCEND_HOME_PATH}/set_env.sh"

# Test cases from PLAN.md
declare -A TEST_CASES
TEST_CASES["TC-01"]="2 1024 4 1280"
TEST_CASES["TC-02"]="1 1 1 1280"
TEST_CASES["TC-03"]="1 1 4 1"
TEST_CASES["TC-04"]="4 2048 8 640"
TEST_CASES["TC-05"]="1 1 4 1279"
TEST_CASES["TC-06"]="1 1 4 12000"
TEST_CASES["TC-07"]="8 512 4 1280"

PASS_COUNT=0
FAIL_COUNT=0
RESULTS=""

echo "========================================="
echo "  expand_kenel_bwd Test Suite"
echo "========================================="

for TC in TC-01 TC-02 TC-03 TC-04 TC-05 TC-06 TC-07; do
    SHAPE="${TEST_CASES[$TC]}"
    echo ""
    echo "--- ${TC}: Shape (${SHAPE}) ---"

    # Generate data (gen_data.py writes to ./input/ and ./output/ relative to CWD)
    python3 scripts/gen_data.py ${SHAPE} 2>&1 | tail -1

    # Remove old output
    rm -f output/output.bin

    # Run kernel (reads ./input/input_x.bin, writes ./output/output.bin)
    ./build/${OP_NAME} ${SHAPE} 2>&1 | grep -E "(Shape|Cores|Tiling)" || true

    # Verify
    if [ -f output/output.bin ]; then
        VERIFY_OUTPUT=$(python3 scripts/verify_result.py output/output.bin output/golden.bin 2>&1)
        echo "$VERIFY_OUTPUT" | grep -E "(MERE|MARE|RESULT)" || true

        if echo "$VERIFY_OUTPUT" | grep -q "PASSED"; then
            echo "  ${TC}: PASSED"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "  ${TC}: FAILED"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "  ${TC}: FAILED (no output file)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "========================================="
echo "  Results: ${PASS_COUNT}/7 PASSED, ${FAIL_COUNT}/7 FAILED"
echo "========================================="

if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
