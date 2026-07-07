export MACA_PATH=${MACA_PATH:-/opt/maca/}

MACA_VISIBLE_DEVICES=0 make clean
MACA_VISIBLE_DEVICES=0 make test_maca

warm_up_count=${1:-5}
test_count=${2:-1000}
exec_mode=${3:-0}

echo "=========================================="
echo "exec_mode = ${exec_mode}"
case ${exec_mode} in
    0) echo "  -> ori + opt: run ori + opt kernel, compare precision" ;;
    1) echo "  -> ori only:    only baseline kernel (*_ori)" ;;
    2) echo "  -> opt only:    only optimized kernel (*_opt)" ;;
    *) echo "  -> unknown mode" ;;
esac
echo "=========================================="

MACA_VISIBLE_DEVICES=0 ./test_maca ${warm_up_count} ${test_count} ${exec_mode}
