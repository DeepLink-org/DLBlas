import os
import glob
import subprocess
import shutil
import sys
from setuptools import setup, Extension

current_path = os.path.abspath(os.path.dirname(__file__))
# SDAA device LTO incorporates source paths into the fatbin layout.
# Normalize checkout paths so identical sources produce identical binaries.
canonical_source_path = "/root/teco-ops-tcdd-20260827-v229-v202-packed-compact-reduce"
reproducible_path_args = [
    f"-ffile-prefix-map={current_path}={canonical_source_path}",
    f"-fdebug-prefix-map={current_path}={canonical_source_path}",
    f"-fmacro-prefix-map={current_path}={canonical_source_path}",
]

# Clean up old .so files before building
for so_file in glob.glob(os.path.join(current_path, "api", "tecoops", "*.so")):
    os.remove(so_file)
    print(f"Removed {so_file}")

# Download third-party dependencies
print("Running tools/deps.py to fetch third-party dependencies...")
sys.path.insert(0, os.path.join(current_path, "tools"))
import deps


def get_version():
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return tag.lstrip("v")
    except subprocess.CalledProcessError:
        try:
            return (
                subprocess.check_output(
                    ["git", "describe", "--tags"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
                .lstrip("v")
            )
        except subprocess.CalledProcessError:
            return "0.0.0"


try:
    from setuptools.command.clean import clean
except ImportError:
    from distutils.command.clean import clean

teco_source_dir = os.path.join(current_path, "teco")
teco_build_dir = os.path.join(current_path, "build", "teco")
teco_lib_dir = os.path.join(teco_build_dir, "lib")


class CMakeClean(clean):
    def run(self):
        clean.run(self)
        if os.path.exists(teco_build_dir):
            shutil.rmtree(teco_build_dir)
            print(f"Removed {teco_build_dir}")


with_torch = os.environ.get("WITH_TORCH", "ON")
with_inference_plugin = os.environ.get("WITH_INFERENCE_PLUGIN", "ON")

if with_torch != "ON" and with_inference_plugin != "ON":
    raise RuntimeError(
        "At least one of WITH_TORCH or WITH_INFERENCE_PLUGIN must be ON."
    )

ext_modules = []
cmdclass = {"clean": CMakeClean}

# --- First pass: Inference plugin ---
if with_inference_plugin == "ON":
    print("WITH_INFERENCE_PLUGIN=ON: Building inference plugin...")

    if os.path.exists(teco_build_dir):
        shutil.rmtree(teco_build_dir)
    os.makedirs(teco_build_dir, exist_ok=True)

    try:
        import tvm

        tvm_path = os.path.dirname(tvm.__file__)
        os.environ["TECO_INFER_PLUGIN_UTIL_PATH"] = tvm_path
    except ImportError as e:
        raise RuntimeError(
            "Failed to import tvm (WITH_INFERENCE_PLUGIN=ON requires it): {e}"
        ) from e

    cmake_cmd = [
        "cmake",
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + teco_lib_dir,
        "-DWITH_INFERENCE_PLUGIN=ON",
        "-S",
        teco_source_dir,
        "-B",
        teco_build_dir,
    ]
    result = subprocess.run(cmake_cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "CMake configure failed (plugin): " + result.stderr.decode(errors="replace")
        )
    result = subprocess.run(["cmake", "--build", teco_build_dir], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "CMake build failed (plugin): " + result.stderr.decode(errors="replace")
        )
    print("libteco_ops_plugin.so and libTecoInferPlugin.so built successfully!")

    teco_ops_plugin_lib = os.path.join(teco_lib_dir, "libteco_ops_plugin.so")
    teco_ops_plugin_dest = os.path.join(
        current_path, "api", "tecoops", "libteco_ops_plugin.so"
    )
    shutil.copy2(teco_ops_plugin_lib, teco_ops_plugin_dest)
    print(f"Copied libteco_ops_plugin.so to api/tecoops/libteco_ops_plugin.so")

    teco_infer_plugin_lib = os.path.join(teco_lib_dir, "libTecoInferPlugin.so")
    teco_infer_plugin_dest = os.path.join(
        current_path, "api", "tecoops", "libTecoInferPlugin.so"
    )
    shutil.copy2(teco_infer_plugin_lib, teco_infer_plugin_dest)
    print(f"Copied libTecoInferPlugin.so to api/tecoops/libTecoInferPlugin.so")
else:
    print("WITH_INFERENCE_PLUGIN=OFF: Skipping inference plugin build")

if with_torch == "ON":
    print("WITH_TORCH=ON: Using TecoExtension for all modules")
    try:
        import torch
        import torch_sdaa
        from torch_sdaa.utils.cpp_extension import TecoExtension, BuildExtension

        torch_sdaa_home = os.path.dirname(torch_sdaa.__file__)
        torch_sdaa_include = os.path.join(torch_sdaa_home, "include")

        torch_cmake_prefix = torch.utils.cmake_prefix_path
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

        # Clean build directory for this pass
        if os.path.exists(teco_build_dir):
            shutil.rmtree(teco_build_dir)
        os.makedirs(teco_build_dir, exist_ok=True)

        # Build: WITH_TORCH=ON (produces libteco_ops.so with torch link flags)
        cmake_cmd = [
            "cmake",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + teco_lib_dir,
            "-DCMAKE_PREFIX_PATH=" + torch_cmake_prefix,
            "-DWITH_TORCH=ON",
            "-DTORCH_SDAA_HOME=" + torch_sdaa_home,
            "-S",
            teco_source_dir,
            "-B",
            teco_build_dir,
        ]
        result = subprocess.run(cmake_cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                "CMake configure failed: " + result.stderr.decode(errors="replace")
            )
        result = subprocess.run(
            ["cmake", "--build", teco_build_dir], capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "CMake build failed: " + result.stderr.decode(errors="replace")
            )
        print("libteco_ops.so built successfully!")

        teco_ops_lib = os.path.join(teco_lib_dir, "libteco_ops.so")
        teco_ops_dest = os.path.join(current_path, "api", "tecoops", "libteco_ops.so")
        shutil.copy2(teco_ops_lib, teco_ops_dest)
        print(f"Copied libteco_ops.so to api/tecoops/")

        # Common link args for both extensions
        common_link_args = [
            "-L" + teco_lib_dir,
            "-lteco_ops",
            "-Wl,-rpath,$$ORIGIN",
            "-L" + torch_lib_dir,
            "-Wl,-rpath," + torch_lib_dir,
        ]

        # Add torch extension module
        ext_modules = [
            TecoExtension(
                name="tecoops._torch_ext",
                sources=["api/torch_ext.cpp"],
                include_dirs=[
                    "teco",
                    torch_sdaa_include,
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-DWITH_TORCH"] + reproducible_path_args,
                },
                extra_link_args=common_link_args,
            )
        ]

        # Override build_ext to use BuildExtension for torch support
        cmdclass = {"build_ext": BuildExtension, "clean": CMakeClean}
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import torch/torch_sdaa (WITH_TORCH=ON requires them): {e}"
        ) from e

setup(
    name="tecoops",
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_dir={"": "api"},
    packages=["tecoops"],
    package_data={
        "tecoops": ["*.so"],
    },
)
