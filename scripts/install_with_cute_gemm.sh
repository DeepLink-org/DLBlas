#!/usr/bin/env bash
# Install DLBlas together with the vendored cute_gemm (CuTe-DSL) operator library.
# Both projects keep independent build configs; this script just runs the two
# installs in order. Run it from anywhere (paths resolve to the repo root).
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install -e .
python -m pip install -e ./dlblas/kernels/cute_gemm
