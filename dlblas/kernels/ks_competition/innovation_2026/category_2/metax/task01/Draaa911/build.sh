#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MACA_PATH="${MACA_PATH:-/opt/maca}"
MXCC="${MXCC:-$MACA_PATH/mxgpu_llvm/bin/mxcc}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3.10}"

cd "$ROOT"
[[ -x "$MXCC" ]] || { echo "missing MACA compiler: $MXCC" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
rm -rf -- "$ROOT/build"
find "$ROOT" -maxdepth 1 -type f \
  \( -name 'libengram.so' -o -name 'engram_bridge_native*.so' \) -delete

"$MXCC" -x maca -offload-arch native -shared -fPIC \
  engram_kernel.cpp -o "$ROOT/libengram.so" --maca-path="$MACA_PATH"

"$PYTHON_BIN" setup.py build_ext --inplace
