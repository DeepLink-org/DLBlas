#!/bin/bash
# ============================================================================
# norm_fn 算子运行脚本
# 编译 → 生成测试数据 → 运行 → 精度验证
#
# 用法：
#   bash run.sh                  # 完整流程（含编译）
#   bash run.sh --skip-build     # 跳过编译，复用已有产物
#   bash run.sh --torch          # 只跑 PyTorch 通路
#   bash run.sh --with-weight    # 生成有权重测试数据
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="norm_fn"

SKIP_BUILD=0
TORCH_ONLY=0
WITH_WEIGHT=""
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
        --with-weight) WITH_WEIGHT="--with-weight" ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"
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
    cmake .. || die "cmake 配置失败"
    make -j4  || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 ==="
cd build
python3 ../scripts/gen_data.py ${WITH_WEIGHT} || die "gen_data.py 执行失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel 运行失败（exit code $?）"
[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在（静默失败）"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "精度验证失败"

echo "=== PyTorch 通路验证 ==="
python3 ../scripts/test_torch.py ${WITH_WEIGHT} \
    || die "PyTorch 通路验证失败"

echo "=== 完成 ==="
exit 0
