"""Build the native Task03 forward bridge against the target torch ABI."""
from __future__ import annotations

import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_LIB = ROOT / "libtask03.so"

if not LOCAL_LIB.is_file():
    raise RuntimeError(
        f"missing device library {LOCAL_LIB}; run build.sh first"
    )

try:
    import torch
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths
except Exception as exc:  # pragma: no cover - target-build failure path
    raise RuntimeError("Task03 bridge requires the target torch/setuptools ABI") from exc


python_include = Path(sysconfig.get_paths()["include"])
python_header = python_include / "Python.h"
torch_root = Path(torch.__file__).resolve().parent
private_header = torch_root / "include" / "torch" / "csrc" / "autograd" / "python_variable.h"
if not python_header.is_file():
    raise RuntimeError(f"Python.h is unavailable: {python_header}")
if not private_header.is_file():
    raise RuntimeError(
        "torch private Tensor bridge header is unavailable: " f"{private_header}"
    )

extension = CppExtension(
    name="task03_native_forward",
    sources=[str(ROOT / "bridge.cpp")],
    include_dirs=[str(python_include), *include_paths()],
    library_dirs=[str(ROOT)],
    libraries=["task03"],
    extra_compile_args={"cxx": ["-O3", "-std=c++17", "-DNDEBUG"]},
    extra_link_args=["-Wl,-rpath,$ORIGIN"],
)

setup(
    name="task03-native-forward",
    version="0.0.0",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
