import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install import install as _install


def _has_nvidia_gpu():
    """Detect an NVIDIA GPU from several independent indicators."""
    override = os.environ.get("DLBLAS_INSTALL_CUTE_GEMM", "auto").lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override in ("1", "true", "yes", "on"):
        return True
    if Path("/proc/driver/nvidia/version").exists():
        return True
    if any(Path("/dev").glob("nvidia[0-9]*")):
        return True
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and "GPU " in result.stdout:
                return True
        except Exception:
            pass
    if shutil.which("lspci"):
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=10
            )
            if "nvidia" in result.stdout.lower():
                return True
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def _install_cute_gemm():
    """Install the vendored cute_gemm on NVIDIA GPUs only."""
    root = Path(__file__).resolve().parent
    target = root / "dlblas" / "kernels" / "cute_gemm"
    if not (target / "pyproject.toml").is_file():
        return
    if not _has_nvidia_gpu():
        print("cute_gemm: no NVIDIA GPU detected; skipping (CUDA-only).")
        return
    print(f"cute_gemm: installing {target}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(target)])


class install(_install):
    def run(self):
        super().run()
        _install_cute_gemm()


def main():
    setup(
        name="dlblas",
        version="0.0.7",
        packages=find_packages(exclude=()),
        cmdclass={"install": install},
    )


if __name__ == "__main__":
    main()
