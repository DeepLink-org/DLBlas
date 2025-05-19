cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(dlblas_extensions LANGUAGES CXX CUDA)

# CUDA by default, can be overridden by using -DDLBLAS_TARGET_DEVICE=... (used by setup.py)
set(DLBLAS_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for dlblas")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${DLBLAS_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")

#
# Try to find python package with an executable that exactly matches
# `DLBLAS_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (DLBLAS_PYTHON_EXECUTABLE)
  find_python_from_executable(${DLBLAS_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set DLBLAS_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()
#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

set(DLBLAS_GPU_LANG "CUDA")
if(DLBLAS_GPU_LANG STREQUAL "CUDA")
  #
  # For cuda we want to be able to control which architectures we compile for on
  # a per-file basis in order to cut down on compile time. So here we extract
  # the set of architectures we want to compile for and remove the from the
  # CMAKE_CUDA_FLAGS so that they are not applied globally.
  #
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
endif()
#
# Query torch for additional GPU compilation flags for the given
# `DLBLAS_GPU_LANG`.
# The final set of arches is stored in `DLBLAS_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(DLBLAS_GPU_FLAGS ${DLBLAS_GPU_LANG})
#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND DLBLAS_GPU_LANG STREQUAL "CUDA")
  list(APPEND DLBLAS_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()
#
# Use FetchContent for C++ dependencies that are compiled as part of dlblas's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# _C extension
#
set(DLBLAS_EXT_SRC
    "csrc/moe/fused_gate.cu"
    "csrc/moe/moe_align_sum_kernels.cu"
    "csrc/activations.cu"
    "csrc/per_token_group_quant_8.cu"
    "csrc/extension.cc"
)
if(DLBLAS_GPU_LANG STREQUAL "CUDA")
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # Set CUTLASS_REVISION manually -- its revision detection doesn't work in this case.
  # Please keep this in sync with FetchContent_Declare line below.
  set(CUTLASS_REVISION "v3.8.0" CACHE STRING "CUTLASS revision to use")

  # Use the specified CUTLASS source directory for compilation if DLBLAS_CUTLASS_SRC_DIR is provided
  if (DEFINED ENV{DLBLAS_CUTLASS_SRC_DIR})
    set(DLBLAS_CUTLASS_SRC_DIR $ENV{DLBLAS_CUTLASS_SRC_DIR})
  endif()

  if(DLBLAS_CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE DLBLAS_CUTLASS_SRC_DIR)
      get_filename_component(DLBLAS_CUTLASS_SRC_DIR "${DLBLAS_CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The DLBLAS_CUTLASS_SRC_DIR is set, using ${DLBLAS_CUTLASS_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${DLBLAS_CUTLASS_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        # Please keep this in sync with CUTLASS_REVISION line above.
        GIT_TAG v3.8.0
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
    )
  endif()
  FetchContent_MakeAvailable(cutlass)
  set_gencode_flags_for_srcs(
    SRCS "${DLBLAS_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
endif()

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  _DLBLAS
  DESTINATION dlblas
  LANGUAGE ${DLBLAS_GPU_LANG}
  SOURCES ${DLBLAS_EXT_SRC}
  COMPILE_FLAGS ${DLBLAS_GPU_FLAGS}
  ARCHITECTURES ${DLBLAS_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_DLBLAS PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
