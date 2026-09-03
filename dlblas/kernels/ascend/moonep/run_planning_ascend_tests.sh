#!/usr/bin/env bash
# Ascend Planning 统一测试入口。
# 默认在选定的真实 NPU 上运行正式功能测试；rank 数由 --npu-ids 自动推导。
# 脚本不安装依赖，也不会使用 NPU 0。

set -Eeuo pipefail

# =============================================================================
# 默认配置
# =============================================================================

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${ROOT_DIR}/../../../.." && pwd)"
MODE="functional"
NPU_IDS="${MOONEP_NPU_IDS:-2,3}"
IMPLS="${MOONEP_PLANNING_IMPLS:-torch_reference_port,triton_reference_semantic}"
CASES="${MOONEP_PLANNING_CASES:-}"
DEDUP_MODES="${MOONEP_PLANNING_DEDUP_MODES:-zero,current,src}"
PERF_DEDUP_MODE="${MOONEP_PERF_DEDUP_MODE:-src}"
PERF_CASE="balanced_epn16"
WARMUP=10
REPEAT=30
MASTER_PORT="${MASTER_PORT:-29631}"
LOG_ROOT="${MOONEP_TEST_LOG_ROOT:-${ROOT_DIR}/logs/test-runs}"
TRITON_CACHE_ROOT="${MOONEP_TRITON_CACHE_DIR:-}"
TRACE="${MOONEP_PLANNING_TRACE:-0}"
NUM_SMS="${MOONEP_NUM_SMS:-}"
TORCHRUN_BIN="${TORCHRUN_BIN:-}"
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-}"

usage() {
    cat <<'EOF'
用法：
  ./run_planning_ascend_tests.sh [模式] [选项]

模式：
  functional   运行低依赖 standalone 功能测试（默认）
  performance  对选中的实现执行统一 correctness + warmup + 性能复测
  all          依次执行功能测试和性能测试

选项：
  --npu-ids IDS       两张或多张物理 NPU，例如 2,3 或 2,3,4,7；禁止包含 0
  --impls NAMES       逗号分隔的实现名称
  --cases NAMES       standalone 功能测试的 case 子集；默认运行全部 standalone case
  --dedup-modes MODES 功能测试模式，可多选（默认 zero,current,src）
  --dedup-mode MODE   性能测试模式（默认 src）
  --perf-case NAME    性能测试 case（默认 balanced_epn16）
  --warmup N          性能测试 warmup 次数（默认 10）
  --repeat N          性能测试正式测量次数（默认 30）
  --num-sms N         覆盖 ctx['num_sms']，例如 910B 使用 24
  --master-port PORT  torchrun master port（默认 29631）
  --log-root DIR      日志根目录（默认 logs/test-runs）
  --triton-cache DIR  复用指定 Triton cache；默认每次运行使用独立 cache
  --torchrun PATH     显式指定 torchrun；默认优先 PATH，其次使用 .venv
  --cann-env PATH     显式指定 CANN set_env.sh；默认自动发现现有安装
  --trace             打开 Planning 分阶段 trace
  -h, --help          显示本帮助

实现名称：
  torch_reference_port
  triton_reference_semantic

示例：
  ./run_planning_ascend_tests.sh functional --npu-ids 2,3
  ./run_planning_ascend_tests.sh functional --impls torch_reference_port
  ./run_planning_ascend_tests.sh performance --warmup 10 --repeat 30
  ./run_planning_ascend_tests.sh performance \
    --npu-ids 2,3,4,7 --impls torch_reference_port \
    --dedup-mode src --num-sms 24
  ./run_planning_ascend_tests.sh all --npu-ids 2,3
EOF
}

# =============================================================================
# 参数解析与静态校验
# =============================================================================

if [[ $# -gt 0 && "${1}" != -* ]]; then
    MODE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --npu-ids)
            NPU_IDS="${2:?--npu-ids 需要参数}"
            shift 2
            ;;
        --impls)
            IMPLS="${2:?--impls 需要参数}"
            shift 2
            ;;
        --cases)
            CASES="${2:?--cases 需要参数}"
            shift 2
            ;;
        --dedup-modes)
            DEDUP_MODES="${2:?--dedup-modes 需要参数}"
            shift 2
            ;;
        --dedup-mode)
            PERF_DEDUP_MODE="${2:?--dedup-mode 需要参数}"
            shift 2
            ;;
        --perf-case)
            PERF_CASE="${2:?--perf-case 需要参数}"
            shift 2
            ;;
        --warmup)
            WARMUP="${2:?--warmup 需要参数}"
            shift 2
            ;;
        --repeat)
            REPEAT="${2:?--repeat 需要参数}"
            shift 2
            ;;
        --num-sms)
            NUM_SMS="${2:?--num-sms 需要参数}"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="${2:?--master-port 需要参数}"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="${2:?--log-root 需要参数}"
            shift 2
            ;;
        --triton-cache)
            TRITON_CACHE_ROOT="${2:?--triton-cache 需要参数}"
            shift 2
            ;;
        --torchrun)
            TORCHRUN_BIN="${2:?--torchrun 需要参数}"
            shift 2
            ;;
        --cann-env)
            CANN_ENV_SCRIPT="${2:?--cann-env 需要参数}"
            shift 2
            ;;
        --trace)
            TRACE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    functional|performance|all) ;;
    *)
        echo "错误：未知模式 ${MODE}" >&2
        usage >&2
        exit 2
        ;;
esac

IFS=',' read -r -a NPU_ARRAY <<< "$NPU_IDS"
NPU_COUNT=${#NPU_ARRAY[@]}
if [[ "$NPU_COUNT" -lt 2 ]]; then
    echo "错误：--npu-ids 至少需要两张卡，当前为 ${NPU_IDS}" >&2
    exit 2
fi
for device_id in "${NPU_ARRAY[@]}"; do
    if [[ ! "$device_id" =~ ^[0-9]+$ ]]; then
        echo "错误：非法 NPU ID ${device_id}" >&2
        exit 2
    fi
    if [[ "$device_id" == "0" ]]; then
        echo "错误：本任务禁止使用 NPU 0" >&2
        exit 2
    fi
done
declare -A SEEN_NPU_IDS=()
for device_id in "${NPU_ARRAY[@]}"; do
    if [[ -n "${SEEN_NPU_IDS[$device_id]:-}" ]]; then
        echo "错误：物理 NPU ID 不能重复：${NPU_IDS}" >&2
        exit 2
    fi
    SEEN_NPU_IDS[$device_id]=1
done

for value_name in WARMUP REPEAT MASTER_PORT; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "错误：${value_name} 必须是非负整数，当前为 ${value}" >&2
        exit 2
    fi
done
if [[ -n "$NUM_SMS" && ! "$NUM_SMS" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：--num-sms 必须是正整数，当前为 ${NUM_SMS}" >&2
    exit 2
fi
if (( REPEAT < 1 )); then
    echo "错误：--repeat 必须大于 0" >&2
    exit 2
fi
if (( MASTER_PORT < 1024 || MASTER_PORT > 65535 )); then
    echo "错误：--master-port 必须位于 1024..65535" >&2
    exit 2
fi

IFS=',' read -r -a IMPL_ARRAY <<< "$IMPLS"
if [[ ${#IMPL_ARRAY[@]} -eq 0 ]]; then
    echo "错误：至少选择一种实现" >&2
    exit 2
fi

module_for_impl() {
    case "$1" in
        torch_reference_port) echo "dlblas.kernels.ascend.moonep.planning_torch_ascend" ;;
        triton_reference_semantic) echo "dlblas.kernels.ascend.moonep.planning_triton_ascend_reference" ;;
        *)
            echo "错误：未知实现 $1" >&2
            return 2
            ;;
    esac
}

for impl in "${IMPL_ARRAY[@]}"; do
    module="$(module_for_impl "$impl")"
    module_path="${module//./\/}"
    if [[ ! -f "${PROJECT_ROOT}/${module_path}.py" ]]; then
        echo "错误：实现 ${impl} 对应的模块不存在：${PROJECT_ROOT}/${module_path}.py" >&2
        exit 2
    fi
done

IFS=',' read -r -a DEDUP_MODE_ARRAY <<< "$DEDUP_MODES"
for dedup_mode in "${DEDUP_MODE_ARRAY[@]}"; do
    case "$dedup_mode" in
        zero|current|src) ;;
        *)
            echo "错误：未知 dedup mode ${dedup_mode}" >&2
            exit 2
            ;;
    esac
done
case "$PERF_DEDUP_MODE" in
    zero|current|src) ;;
    *)
        echo "错误：未知性能 dedup mode ${PERF_DEDUP_MODE}" >&2
        exit 2
        ;;
esac

# =============================================================================
# 运行环境与日志
# =============================================================================

if [[ -z "$CANN_ENV_SCRIPT" ]]; then
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
        CANN_ENV_SCRIPT="${ASCEND_HOME_PATH}/set_env.sh"
    elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
        CANN_ENV_SCRIPT="/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    elif [[ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
        CANN_ENV_SCRIPT="/usr/local/Ascend/cann-9.1.0/set_env.sh"
    fi
fi
if [[ -z "$CANN_ENV_SCRIPT" || ! -f "$CANN_ENV_SCRIPT" ]]; then
    echo "错误：找不到 CANN set_env.sh；请使用 --cann-env PATH" >&2
    exit 127
fi
# shellcheck disable=SC1090
source "$CANN_ENV_SCRIPT"

if [[ -z "$TORCHRUN_BIN" ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/torchrun" ]]; then
        TORCHRUN_BIN="${PROJECT_ROOT}/.venv/bin/torchrun"
    elif command -v torchrun >/dev/null 2>&1; then
        TORCHRUN_BIN="$(command -v torchrun)"
    fi
elif [[ "$TORCHRUN_BIN" != */* ]]; then
    TORCHRUN_BIN="$(command -v "$TORCHRUN_BIN" || true)"
fi
if [[ -z "$TORCHRUN_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
    echo "错误：找不到可执行的 torchrun；请激活环境或使用 --torchrun PATH" >&2
    exit 127
fi

required_files=(
    "run_reference_optimized_round.py"
    "test_planning_triton_ascend_standalone.py"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${ROOT_DIR}/${required_file}" ]]; then
        echo "错误：缺少 ${ROOT_DIR}/${required_file}" >&2
        exit 2
    fi
done

RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODE}"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
if [[ -z "$TRITON_CACHE_ROOT" ]]; then
    TRITON_CACHE_ROOT="${RUN_DIR}/triton-cache"
fi
mkdir -p "$RUN_DIR" "$TRITON_CACHE_ROOT"

export MOONEP_NPU_IDS="$NPU_IDS"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS"
export MOONEP_PLANNING_IMPLS="$IMPLS"
export MOONEP_PLANNING_DEDUP_MODES="$DEDUP_MODES"
export MOONEP_PLANNING_TRACE="$TRACE"
export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "$CASES" ]]; then
    export MOONEP_PLANNING_CASES="$CASES"
else
    unset MOONEP_PLANNING_CASES || true
fi

cd "$PROJECT_ROOT"

TORCHRUN=(
    "$TORCHRUN_BIN"
    --nnodes=1
    --nproc_per_node="$NPU_COUNT"
    --master_addr=127.0.0.1
    --master_port="$MASTER_PORT"
)

run_logged() {
    local label="$1"
    shift
    local log_file="${RUN_DIR}/${label}.log"
    echo "[$(date '+%F %T')] START ${label}"
    echo "命令：$*"
    "$@" 2>&1 | tee "$log_file"
    echo "[$(date '+%F %T')] PASS  ${label}"
}

# =============================================================================
# 功能与性能测试
# =============================================================================

run_functional() {
    run_logged \
        "functional_standalone" \
        "${TORCHRUN[@]}" -m pytest -s -q \
        "${ROOT_DIR}/test_planning_triton_ascend_standalone.py"
}

run_performance() {
    local impl
    local module
    local -a num_sms_args=()
    if [[ -n "$NUM_SMS" ]]; then
        num_sms_args+=(--num-sms "$NUM_SMS")
    fi
    for impl in "${IMPL_ARRAY[@]}"; do
        module="$(module_for_impl "$impl")"
        run_logged \
            "performance_${impl}" \
            "${TORCHRUN[@]}" "${ROOT_DIR}/run_reference_optimized_round.py" \
            --module "$module" \
            --case "$PERF_CASE" \
            --dedup-mode "$PERF_DEDUP_MODE" \
            --warmup "$WARMUP" \
            --repeat "$REPEAT" \
            "${num_sms_args[@]}"
    done
}

# =============================================================================
# 调度与最终状态
# =============================================================================

echo "Planning Ascend 测试开始"
echo "  mode=${MODE}"
echo "  npu_ids=${NPU_IDS}"
echo "  ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  nproc_per_node=${NPU_COUNT}"
echo "  implementations=${IMPLS}"
echo "  functional_cases=${CASES:-all}"
echo "  functional_dedup_modes=${DEDUP_MODES}"
echo "  performance_case=${PERF_CASE} warmup=${WARMUP} repeat=${REPEAT}"
echo "  performance_dedup_mode=${PERF_DEDUP_MODE}"
echo "  num_sms=${NUM_SMS:-case-default}"
echo "  torchrun=${TORCHRUN_BIN}"
echo "  cann_env=${CANN_ENV_SCRIPT}"
echo "  logs=${RUN_DIR}"
echo "  triton_cache=${TRITON_CACHE_DIR}"

case "$MODE" in
    functional)
        run_functional
        ;;
    performance)
        run_performance
        ;;
    all)
        run_functional
        run_performance
        ;;
esac

echo "Planning Ascend 测试全部通过，日志目录：${RUN_DIR}"
