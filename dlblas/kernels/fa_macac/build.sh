#!/bin/bash
# build.sh — compile fa_my_builtin (fully-builtin flash-attn fwd, hdim128).
#
# Self-locating & portable: no hardcoded absolute paths.
#   - Our own sources are found relative to this script (./src).
#   - The MACA system SDK is located via the MACA_PATH env var (must point to
#     the official MACA install root, i.e. the dir containing include/,
#     tools/, and lib/).
set -euo pipefail

: "${MACA_PATH:?Error: set MACA_PATH to your MACA SDK install root (the dir containing include/, tools/, lib/)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUT="$SCRIPT_DIR/fa_my_builtin"

CXX="$MACA_PATH/tools/cu-bridge/bin/cucc"

D="-DFA_DTYPE_FP16 -DHDIM_128 -DHDIM_CONFIG=128 -DUSE_MACA -DXCORE1000 -D__FAST_HALF_CVT__ -D__MERGE_LDS_B64"
I="-I$SRC_DIR -I$MACA_PATH/include -I$MACA_PATH/tools/cu-bridge/include"
F="-w -Wno-format -x maca --compiler-options -fPIC -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -fno-strict-aliasing -mllvm -metaxgpu-inlinescope=50 -mllvm -metaxgpu-disable-bsm-offset=0 -Xclang -menable-no-nans -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -mllvm -metaxgpu-enable-promote-kernel-arguments -mllvm -metaxgpu-const-KArgs-LDU=1 -mllvm -metaxgpu-enable-gvn-loadclobber=false -mllvm -metaxgpu-enable-unorder-dispatch -mllvm -metaxgpu-disable-early-vector-combine=true -mllvm -metaxgpu-llvm19-inline-max-bb=2000 -mllvm -metaxgpu-preisel-simplifycfg=true --offload-arch=xcore1000"
L="-L$MACA_PATH/lib -lmcruntime -lmxc-runtime64 -lcurand -lmcToolsExt -lcudart"

echo "Compiling fa_my_builtin (MACA_PATH=$MACA_PATH, SRC=$SRC_DIR)..."
"$CXX" $D $I $F $L -o "$OUT" "$SRC_DIR/fa_my.cu" 2>&1 | tail -50
echo "Exit: ${PIPESTATUS[0]}"

if [ -f "$OUT" ]; then
  echo "BUILD_OK -> $OUT"
else
  echo "BUILD_FAIL"; exit 1
fi
