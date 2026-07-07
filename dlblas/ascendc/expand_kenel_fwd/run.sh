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
# expand_kenel_fwd 一键运行脚本
# 生成数据 → 编译 → 运行 Kernel → 精度验证
#
# 用法：
#   bash run.sh [B] [S] [H] [M] [dtype]
#   bash run.sh --skip-build [B] [S] [H] [M] [dtype]
#   bash run.sh --torch                      (仅 PyTorch 通路)
#
#   默认: B=1 S=1024 H=1280 M=4 dtype=fp16
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="expand_kenel_fwd"
SKIP_BUILD=0
TORCH_ONLY=0

# 解析参数
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --torch)
            TORCH_ONLY=1
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 默认参数
B="${POSITIONAL_ARGS[0]:-1}"
S="${POSITIONAL_ARGS[1]:-1024}"
H="${POSITIONAL_ARGS[2]:-1280}"
M="${POSITIONAL_ARGS[3]:-4}"
DTYPE="${POSITIONAL_ARGS[4]:-fp16}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== expand_kenel_fwd ==="
echo "  Shape: B=${B} S=${S} H=${H} M=${M} dtype=${DTYPE}"

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先配置 CANN 环境"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch 通路验证 ==="
    cd build
    python3 ../scripts/test_torch.py || die "PyTorch 通路验证失败"
    echo "=== 完成 ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在，请先完整编译"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release || die "cmake 配置失败"
    make -j4  || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 + 运行 Kernel ==="
cd build
python3 ../scripts/gen_data.py "${B}" "${S}" "${H}" "${M}" "${DTYPE}" || die "gen_data.py 执行失败"

# 确定 dtype flag: fp16→0, fp32→1
if [ "${DTYPE}" == "fp32" ]; then
    DTYPE_FLAG=1
else
    DTYPE_FLAG=0
fi

rm -f output/output.bin
"./${OP_NAME}" "${B}" "${S}" "${H}" "${M}" "${DTYPE_FLAG}" || die "Kernel 运行失败 (exit code $?)"
[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在（静默失败）"

echo "=== [4/4] 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin "${DTYPE}" \
    || die "精度验证失败"

echo "=== PyTorch 通路验证 ==="
python3 ../scripts/test_torch.py \
    || die "PyTorch 通路验证失败"

echo "=== 完成 ==="
exit 0
