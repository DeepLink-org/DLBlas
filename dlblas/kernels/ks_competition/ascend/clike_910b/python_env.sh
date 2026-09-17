find_dlblas_python() {
    local candidate resolved
    local -a candidates=()

    if [[ -n "${DLBLAS_PYTHON_EXECUTABLE:-}" ]]; then
        candidates+=("${DLBLAS_PYTHON_EXECUTABLE}")
    else
        candidates+=(python3 python)
        for candidate in /usr/local/python*/bin/python3 /usr/local/bin/python3; do
            candidates+=("${candidate}")
        done
    fi

    for candidate in "${candidates[@]}"; do
        if [[ "${candidate}" == */* ]]; then
            resolved="${candidate}"
        else
            resolved="$(command -v "${candidate}" 2>/dev/null || true)"
        fi
        if [[ -z "${resolved}" || ! -x "${resolved}" ]]; then
            continue
        fi
        if "${resolved}" -c "import torch, torch_npu" >/dev/null 2>&1; then
            "${resolved}" -c "import os, sys; print(os.path.realpath(sys.executable))"
            return 0
        fi
    done

    echo "No Python interpreter with both torch and torch_npu was found." >&2
    echo "Set DLBLAS_PYTHON_EXECUTABLE to the intended Python executable." >&2
    return 1
}
