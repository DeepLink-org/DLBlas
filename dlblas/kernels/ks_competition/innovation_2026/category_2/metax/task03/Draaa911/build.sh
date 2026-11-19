#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MACA_PATH="${MACA_PATH:-/opt/maca}"
MXCC="${MXCC:-$MACA_PATH/mxgpu_llvm/bin/mxcc}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MCTLASS_INCLUDE="${MCTLASS_INCLUDE:-$MACA_PATH/include}"
BUILD_DIR="$ROOT/build-task03"
OBJECT="$BUILD_DIR/task03_kernel.o"
LIB="$ROOT/libtask03.so"

cd "$ROOT"
[[ -x "$MXCC" ]] || { echo "missing MACA compiler: $MXCC" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$MCTLASS_INCLUDE" ]] || { echo "missing MACA include root: $MCTLASS_INCLUDE" >&2; exit 2; }

rm -rf -- "$BUILD_DIR" "$ROOT/build"
find "$ROOT" -maxdepth 1 -type f \( -name 'libtask03.so' -o -name 'task03_native_forward*.so' \) -delete
mkdir -- "$BUILD_DIR"

"$MXCC" -x maca -offload-arch native -resource-usage -c -fPIC \
  -I"$MCTLASS_INCLUDE" "$ROOT/task03_kernel.cpp" -o "$OBJECT" \
  --maca-path="$MACA_PATH"
"$MXCC" -shared "$OBJECT" -o "$LIB" --maca-path="$MACA_PATH"
"$PYTHON_BIN" setup.py build_ext --inplace
