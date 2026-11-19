#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ${1-} != --remote-gpu || -z ${3-} ]]; then
    printf '%s\n' '仅在远程 Linux GPU 主机运行。用法：bash scripts/reproduce.sh --remote-gpu <T1|T2|T3|T4|all> <新的工作目录>' >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    printf '%s\n' '复现需要远程 Linux GPU 环境。' >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
selection="$2"
work_dir="$3"
bash "${SCRIPT_DIR}/prepare.sh" "${selection}" "${work_dir}"
work_dir="$(cd -- "${work_dir}" && pwd -P)"
ntops_dir="${work_dir}/ntops"
ninetoothed_dir="${work_dir}/ninetoothed"
mkdir -- "${work_dir}/results"

run_task() {
    local task="$1" output_dir="${work_dir}/results/$1" cache_dir="${work_dir}/caches/$1"
    mkdir -p -- "${cache_dir}/triton" "${cache_dir}/xdg"
    printf '开始评测 %s，结果目录：%s\n' "${task}" "${output_dir}"
    (
        cd -- "${ntops_dir}"
        export TRITON_CACHE_DIR="${cache_dir}/triton"
        export XDG_CACHE_HOME="${cache_dir}/xdg"
        case "${task}" in
            T1) bash scripts/run_kernelswift_t1.sh --ninetoothed-dir "${ninetoothed_dir}" --output-dir "${output_dir}" --mode all ;;
            T2) bash scripts/run_kernelswift_t2.sh --ninetoothed-dir "${ninetoothed_dir}" --output-dir "${output_dir}" --baseline vllm_portable --mode all ;;
            T3) bash scripts/run_kernelswift_t3.sh --ninetoothed-dir "${ninetoothed_dir}" --output-dir "${output_dir}" --mode all ;;
            T4) NINETOOTHED_ROOT="${ninetoothed_dir}" bash benchmarks/run_kernelswift_t4.sh "${output_dir}" ;;
        esac
    )
}

if [[ ${selection} == all ]]; then
    for task in T1 T2 T3 T4; do
        run_task "${task}"
    done
else
    run_task "${selection}"
fi
