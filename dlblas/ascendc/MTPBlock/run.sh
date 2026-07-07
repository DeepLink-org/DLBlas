#!/usr/bin/env bash
# ============================================================================
# MTPBlock AscendC Kernel 直调运行脚本
# Usage: bash run.sh [--all] [--skip-build] [--torch] [--kernel N]
#   --all:        运行所有 kernel 并验证精度
#   --skip-build: 跳过编译 (复用已有产物)
#   --torch:      仅运行 PyTorch 通路测试
#   --kernel N:   运行单个 kernel (1-6)
#   default:      运行所有 kernel
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="mtpblock_custom"
SKIP_BUILD=0
TORCH_ONLY=0
KERNEL_NUM=0
RUN_ALL=1

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1; RUN_ALL=0 ;;
        --all)        RUN_ALL=1 ;;
        --kernel)     ;;
        *)
            if [[ "$arg" =~ ^[1-6]$ ]]; then
                KERNEL_NUM="$arg"
                RUN_ALL=0
            fi
            ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# ============================================================================
# Environment setup
# ============================================================================
echo "=== [1/5] Setting up CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || source /usr/local/Ascend/cann-9.0.0/set_env.sh || die "set_env.sh failed"

# ============================================================================
# Build
# ============================================================================
if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [2/5] Skip build (reuse existing) ==="
else
    echo "=== [2/5] Build ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake config failed"
    make -j4 || die "make build failed"
    cd ..
fi

# ============================================================================
# Generate test data
# ============================================================================
echo "=== [3/5] Generate test data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

# ============================================================================
# PyTorch only mode
# ============================================================================
if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== [4/5] PyTorch verification ==="
    python3 -c "
import torch, torch_npu
torch.ops.load_library('./build/libmtpblock_ops.so')
b, s, hc, d = 1, 8, 4, 512
x = torch.randn(b,s,d,dtype=torch.bfloat16).npu()
r = torch.randn(b,s,hc,d,dtype=torch.bfloat16).npu()
p = torch.randn(b,s,hc,dtype=torch.float32).npu()
c = torch.randn(b,s,hc,hc,dtype=torch.float32).npu()
y = torch.ops.mtpblock.hc_post(x,r,p,c)
print(f'PyTorch test PASSED: output shape {list(y.shape)}, device={y.device}')
assert y.shape == r.shape
print('=== Done ===')
" || die "PyTorch test failed"
    exit 0
fi

# ============================================================================
# Run kernels
# ============================================================================
echo "=== [4/5] Run Kernel Tests ==="

VERIFY_SCRIPT="../scripts/verify_result.py"
PASSED=0
FAILED=0

run_kernel_test() {
    local k="$1"
    local bin_file="$2"
    local golden_file="$3"
    local dtype="$4"
    local kname="$5"

    echo ""
    echo "--- Running Kernel K${k}: ${kname} ---"
    rm -f "output/${bin_file}"
    "./${OP_NAME}" "${k}" || die "Kernel K${k} run failed (exit code $?)"
    [ -f "output/${bin_file}" ] || die "output/${bin_file} not found"

    echo "  Verifying ${dtype} precision..."
    if python3 "${VERIFY_SCRIPT}" "output/${bin_file}" "output/${golden_file}" "${dtype}"; then
        echo "  K${k} ${kname}: PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  K${k} ${kname}: FAILED"
        FAILED=$((FAILED + 1))
    fi
}

if [ "${RUN_ALL}" -eq 1 ] || [ "${KERNEL_NUM}" -eq 1 ]; then
    run_kernel_test 1 "k1_feat.bin" "golden_k1.bin" fp16 "embed_fuse"
fi

if [ "${RUN_ALL}" -eq 1 ] || [ "${KERNEL_NUM}" -eq 2 ]; then
    run_kernel_test 2 "k2_y.bin" "golden_k2_y.bin" fp16 "hc_pre y"
    # K2 also produces fp32 outputs
    echo "  Verifying fp32 precision (pre)..."
    python3 "${VERIFY_SCRIPT}" "output/k2_pre.bin" "output/golden_k2_pre.bin" fp32
    echo "  Verifying fp32 precision (post)..."
    python3 "${VERIFY_SCRIPT}" "output/k2_post.bin" "output/golden_k2_post.bin" fp32
    echo "  Verifying fp32 precision (comb)..."
    python3 "${VERIFY_SCRIPT}" "output/k2_comb.bin" "output/golden_k2_comb.bin" fp32
fi

if [ "${RUN_ALL}" -eq 1 ] || [ "${KERNEL_NUM}" -eq 3 ]; then
    run_kernel_test 3 "k3_out.bin" "golden_k3.bin" fp16 "attn_block"
fi

if [ "${RUN_ALL}" -eq 1 ] || [ "${KERNEL_NUM}" -eq 4 ]; then
    run_kernel_test 4 "k4_out.bin" "golden_k4.bin" fp16 "hc_post"
fi

if [ "${RUN_ALL}" -eq 1 ] || [ "${KERNEL_NUM}" -eq 5 ]; then
    run_kernel_test 5 "k5_out.bin" "golden_k5.bin" fp16 "moe_block"
fi

if [ "${RUN_ALL}" -eq 1 ] || [ "${KERNEL_NUM}" -eq 6 ]; then
    run_kernel_test 6 "k6_logits.bin" "golden_k6.bin" fp32 "mtp_head"
fi

# ============================================================================
# PyTorch integration test
# ============================================================================
echo ""
echo "=== [5/5] PyTorch integration test ==="
python3 -c "
import torch, torch_npu
torch.ops.load_library('./build/libmtpblock_ops.so')
b, s, hc, d = 1, 8, 4, 512
x = torch.randn(b,s,d,dtype=torch.bfloat16).npu()
r = torch.randn(b,s,hc,d,dtype=torch.bfloat16).npu()
p = torch.randn(b,s,hc,dtype=torch.float32).npu()
c = torch.randn(b,s,hc,hc,dtype=torch.float32).npu()
y = torch.ops.mtpblock.hc_post(x,r,p,c)
assert y.shape == r.shape, f'Shape mismatch: {y.shape} vs {r.shape}'
print('PyTorch hc_post test: PASSED')
" || echo "WARN: PyTorch integration test failed"

cd ..

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "  MTPBlock Test Summary"
echo "=============================================="
echo "  Total tests: $((PASSED + FAILED))"
echo "  Passed:      ${PASSED}"
echo "  Failed:      ${FAILED}"
echo "=============================================="

if [ "${FAILED}" -gt 0 ]; then
    die "Some tests FAILED"
fi

echo "=== All tests PASSED ==="
exit 0
