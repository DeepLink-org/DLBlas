# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# Ascend C Kernel Direct Invoke Runner - engram_fused_weight
# Build -> Generate test data -> Run -> Verify accuracy
#
# Usage:
#   bash run.sh              # Full flow (with build)
#   bash run.sh --skip-build # Skip build, reuse existing artifacts
#   bash run.sh --torch      # PyTorch path only
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="engram_fused_weight"

SKIP_BUILD=0
TORCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setup CANN Environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set, please configure CANN environment"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch Path Verification ==="
    cd build
    python3 ../scripts/test_torch.py || die "PyTorch path verification failed"
    echo "=== Done ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} does not exist, please build first"
    echo "=== [2/4] Skip Build (reusing existing artifacts) ==="
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
[ -f output/output.bin ] || die "output.bin not found after kernel run (silent failure)"

echo "=== Accuracy Verification ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "Accuracy verification failed"

echo "=== PyTorch Path Verification ==="
python3 ../scripts/test_torch.py \
    || die "PyTorch path verification failed"

echo "=== Done ==="
exit 0
