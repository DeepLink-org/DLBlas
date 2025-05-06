# modify from vllm
import logging
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME

ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


def is_ninja_available() -> bool:
    return which('ninja') is not None


def is_sccache_available() -> bool:
    return which('sccache') is not None


def is_ccache_available() -> bool:
    return which('ccache') is not None


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        return os.cpu_count(), 1

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        cfg = 'Debug' if self.debug else 'RelWithDebInfo'
        # cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DDLBLAS_TARGET_DEVICE={}'.format('cuda'),
        ]

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DDLBLAS_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DDLBLAS_PYTHON_PATH={}'.format(':'.join(sys.path))]
        print(cmake_args)

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, '.deps')
        fc_base_dir = os.environ.get('FETCHCONTENT_BASE_DIR', fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        # if _is_cuda():
        cmake_args += [f'-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc']
        subprocess.check_call(['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args], cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix('dlblas.')

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            '--build',
            '.',
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(['cmake', *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            # We assume only the final component of extension prefix is added by
            # CMake, this is currently true for current extensions but may not
            # always be the case.
            prefix = outdir
            if '.' in ext.name:
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = ['cmake', '--install', '.', '--prefix', prefix, '--component', target_name(ext.name)]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        super().run()


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / 'requirements'

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split('\n')
        resolved_requirements = []
        for line in requirements:
            if line.startswith('-r '):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith('--') and not line.startswith('#') and line.strip() != '':
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements('cuda.txt')
    return requirements


ext_modules = []
ext_modules.append(CMakeExtension(name='dlblas._DLBLAS'))

setup(
    name='dlblas',
    version='0.0.2',
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    # package_dir={"": "dlblas"},
    cmdclass={'build_ext': cmake_build_ext},
    packages=find_packages(exclude=()),
)
