from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


root = Path(__file__).resolve().parent
extension = CppExtension(
    name="engram_bridge_native",
    sources=[str(root / "bridge.cpp")],
    library_dirs=[str(root)],
    libraries=["engram"],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
    extra_link_args=["-Wl,-rpath,$ORIGIN"],
)

setup(
    name="engram-bridge-native-forward",
    version="0.0.0",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
