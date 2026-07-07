#!/bin/bash
# Run Sinkhorn Normalize aclnn example
set -e

BASE_PATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASE_PATH}/../build"

usage() {
    echo "Usage: ./run.sh [--eager]"
    echo "  --eager   Run aclnn eager mode example (default)"
}

MODE="eager"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eager) MODE="eager"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Find the test binary
TEST_BIN="${BUILD_PATH}/examples/test_aclnn_sinkhorn_normalize"
if [ ! -f "$TEST_BIN" ]; then
    echo "Test binary not found: $TEST_BIN"
    echo "Please build first: cd .. && bash build.sh --soc=ascend910b"
    exit 1
fi

# Set library path
export LD_LIBRARY_PATH="${BUILD_PATH}/op_host:${BUILD_PATH}/op_api:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/aarch64-linux/lib64:${LD_LIBRARY_PATH}"

echo "Running Sinkhorn Normalize example..."
${TEST_BIN}
echo "Done!"
