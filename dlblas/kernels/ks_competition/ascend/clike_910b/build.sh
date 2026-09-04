#!/usr/bin/env bash
# Keep shell scripts LF-only; Bash treats a trailing CR as part of option names.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
ASCEND_CMAKE="${ASCEND_ROOT}/aarch64-linux/tikcpp/ascendc_kernel_cmake"

source "${PROJECT_DIR}/python_env.sh"
PYTHON_BIN="$(find_dlblas_python)"
echo "Using Python: ${PYTHON_BIN}"

cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${ASCEND_CMAKE}" \
    -DPython3_EXECUTABLE:FILEPATH="${PYTHON_BIN}"
cmake --build "${PROJECT_DIR}/build" --parallel "${BUILD_JOBS:-64}"
