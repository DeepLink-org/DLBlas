export MACA_PATH=${MACA_PATH:-/opt/maca/}

warm_up_count=${1:-0}
test_count=${2:-1}
exec_mode=${3:-0}

# ---- 选一张空闲 GPU ----------------------------------------------------------
pick_free_gpu() {
    local total busy id candidates lock_dir lock_fd
    total=$(mx-smi -L 2>/dev/null | grep -cE '^GPU#[0-9]+' || echo 0)
    if [ "$total" -le 0 ]; then total=8; fi

    busy=$(mx-smi 2>/dev/null \
        | awk '/^\| Process:/{p=1; next}
               p && /^\|[[:space:]]+[0-9]+[[:space:]]+[0-9]+/ {print $2}' \
        | sort -u)

    candidates=""
    for ((id = 0; id < total; id++)); do
        if ! echo "$busy" | grep -qx "$id"; then
            candidates="$candidates $id"
        fi
    done
    [ -z "$candidates" ] && candidates=$(seq 0 $((total - 1)))

    lock_dir=/tmp
    for id in $candidates; do
        exec 9>"$lock_dir/maca_gpu_${id}.lock"
        if flock -n 9; then
            echo "$id"
            return 0
        fi
        exec 9>&-
    done
    return 1
}

if [ -z "${MACA_VISIBLE_DEVICES:-}" ]; then
    GPU_ID=$(pick_free_gpu)
    if [ -z "$GPU_ID" ]; then
        echo "[run.sh] ERROR: 找不到可用 GPU（全部被占用且锁竞争失败）" >&2
        exit 1
    fi
    export MACA_VISIBLE_DEVICES=$GPU_ID
    echo "[run.sh] 自动选卡 MACA_VISIBLE_DEVICES=$GPU_ID（已加文件锁，整个 run.sh 期间独占）"
else
    echo "[run.sh] 使用用户指定 MACA_VISIBLE_DEVICES=$MACA_VISIBLE_DEVICES（未加锁）"
fi
# ----------------------------------------------------------------------------

make clean
make test_maca

echo "=========================================="
echo "exec_mode = ${exec_mode}"
case ${exec_mode} in
    0) echo "  -> ori + opt模式: 运行 ori + opt kernel，比较输出精度" ;;
    1) echo "  -> 仅 ori 模式 : 仅运行原始 kernel (*_ori)" ;;
    2) echo "  -> 仅 opt 模式 : 仅运行优化 kernel (*_opt)" ;;
    *) echo "  -> 未知模式" ;;
esac
echo "=========================================="

./test_maca ${warm_up_count} ${test_count} ${exec_mode}

###   bash run.sh >log.txt 2>&1
