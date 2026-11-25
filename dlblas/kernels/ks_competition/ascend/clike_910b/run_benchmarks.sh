#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${SCRIPT_DIR}/python_env.sh"
PYTHON_BIN="$(find_dlblas_python)"

cd "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
echo "Using Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" -u benchmarks/ks/auto_bench.py \
    --v0_file dlblas/kernels/ks_competition/torch/sparse_attention.py \
    --v1_file dlblas/kernels/ks_competition/ascend/clike_sparse_attention.py \
    --warmup 100 \
    --repeat 1000

"${PYTHON_BIN}" -u benchmarks/ks/auto_bench.py \
    --v0_file dlblas/kernels/ks_competition/torch/indexer.py \
    --v1_file dlblas/kernels/ks_competition/ascend/clike_indexer.py \
    --warmup 100 \
    --repeat 1000

"${PYTHON_BIN}" -u benchmarks/ks/auto_bench.py \
    --v0_file dlblas/kernels/ks_competition/torch/sinkhorn.py \
    --v1_file dlblas/kernels/ks_competition/ascend/clike_sinkhorn.py \
    --warmup 100 \
    --repeat 10000
