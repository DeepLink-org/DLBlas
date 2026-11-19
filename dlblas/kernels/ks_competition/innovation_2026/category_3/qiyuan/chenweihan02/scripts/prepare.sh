#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 2 || -z ${2-} ]]; then
    fail '用法：bash scripts/prepare.sh <T1|T2|T3|T4|all> <新的工作目录>'
fi

selection="$1"
case "${selection}" in
    T1|T2|T3|T4|all) ;;
    *) fail "未知赛题：${selection}" ;;
esac

work_dir="$2"
mkdir -- "${work_dir}" || fail "无法创建新目录（父目录须存在，目标不得存在）：${work_dir}"
work_dir="$(cd -- "${work_dir}" && pwd -P)"

locked_value() {
    awk -v key="$1" '$1 == key { print $2; count++ } END { if (count != 1) exit 1 }' \
        "${SUBMISSION_DIR}/sources.lock"
}

fetch_base() {
    local name="$1" repo="$2" url base
    url="$(locked_value "${name}.url")"
    base="$(locked_value "${name}.base")"
    git init --quiet --template= "${repo}"
    git -C "${repo}" config --local core.autocrlf false
    git -C "${repo}" config --local core.eol lf
    git -C "${repo}" -c gc.auto=0 fetch --quiet --no-tags --no-auto-gc --depth=1 "${url}" "${base}"
    [[ $(git -C "${repo}" rev-parse FETCH_HEAD) == "${base}" ]] || fail "${name}：上游基线不匹配"
    git -C "${repo}" update-ref --no-deref HEAD "${base}"
    git -C "${repo}" read-tree "${base}"
}

verify_and_checkout() {
    local repo="$1" key="$2" expected actual
    expected="$(locked_value "${key}")"
    actual="$(git -C "${repo}" write-tree)"
    [[ ${actual} == "${expected}" ]] || fail "${key}：期望 ${expected}，实际 ${actual}"
    git -C "${repo}" checkout-index --all --index
    git -C "${repo}" diff-files --quiet || fail "${key}：工作区与已校验的索引不一致"
    printf '%s %s\n' "${key}" "${actual}"
}

ntops_dir="${work_dir}/ntops"
ninetoothed_dir="${work_dir}/ninetoothed"

fetch_base ntops "${ntops_dir}"
if [[ ${selection} == all ]]; then
    for task in T1 T2 T3 T4; do
        git -C "${ntops_dir}" apply --cached --whitespace=nowarn \
            --exclude='src/ntops/kernels/__init__.py' \
            --exclude='src/ntops/torch/__init__.py' \
            "${SUBMISSION_DIR}/tasks/${task}/ntops.patch"
    done
    git -C "${ntops_dir}" apply --cached --whitespace=nowarn "${SUBMISSION_DIR}/shared/ntops-all.patch"
else
    git -C "${ntops_dir}" apply --cached --whitespace=nowarn "${SUBMISSION_DIR}/tasks/${selection}/ntops.patch"
fi
verify_and_checkout "${ntops_dir}" "ntops.${selection}.tree"

fetch_base ninetoothed "${ninetoothed_dir}"
git -C "${ninetoothed_dir}" apply --cached --whitespace=nowarn "${SUBMISSION_DIR}/shared/ninetoothed-215.patch"
verify_and_checkout "${ninetoothed_dir}" ninetoothed.tree
