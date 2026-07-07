export MACA_PATH=${MACA_PATH:-/opt/maca/}

MACA_VISIBLE_DEVICES=0 make clean
MACA_VISIBLE_DEVICES=0 make test_maca

warm_up_count=${1:-5}
test_count=${2:-1000}
exec_mode=${3:-0}

echo "=========================================="
echo "exec_mode = ${exec_mode}"
case ${exec_mode} in
    0) echo "  -> ori + opt: 运行 ori + opt kernel，比较精度" ;;
    1) echo "  -> 仅 ori:    仅运行 baseline kernel (*_ori)" ;;
    2) echo "  -> 仅 opt:    仅运行优化 kernel (*_opt)" ;;
    *) echo "  -> 未知模式" ;;
esac
echo "=========================================="

MACA_VISIBLE_DEVICES=0 ./test_maca ${warm_up_count} ${test_count} ${exec_mode}
