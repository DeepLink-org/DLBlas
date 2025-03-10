import os
import subprocess
from pathlib import Path
from setuptools import setup

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

subprocess.check_call(f"nvcc {cwd}/dlblas/kernels/cutlass/fp8_convert.cu -I{cwd}/dlblas/third_party/cutlass/include -shared --compiler-options=-fPIC -o {cwd}/dlblas/kernels/cutlass/fp8_convert.so", shell=True)

setup(
    name="dlblas",
    version="0.0.1",
    package_data = {
        '': ['*.so'],
    }
    # install_requires=[
    # ],
)
