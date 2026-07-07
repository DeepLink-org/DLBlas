# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt - sparse_attn Ascend C Kernel 直调工程
# ----------------------------------------------------------------------------------------------------------
#
# Two targets:
#   1. sparse_attn (executable) - direct-invoke verification
#   2. libsparse_attn_ops.so - PyTorch TORCH_LIBRARY module

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(sparse_attn LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ============================================================================
# Target 1: Executable (direct-invoke)
# ============================================================================
add_executable(sparse_attn op_host/sparse_attn_main.asc)

target_link_libraries(sparse_attn PRIVATE
    tiling_api register platform unified_dlog dl m graph_base)

target_include_directories(sparse_attn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host)

# DAV_2201 Vector architecture for Ascend 910B2
target_compile_options(sparse_attn PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ============================================================================
# Target 2: TORCH_LIBRARY module
# ============================================================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_NPU_PATH)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(sparse_attn_ops SHARED
    op_kernel/sparse_attn_kernel.asc
    op_extension/sparse_attn_torch.cpp
    op_extension/register.cpp)

target_include_directories(sparse_attn_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR})

target_link_libraries(sparse_attn_ops PRIVATE
    torch_npu tiling_api register platform unified_dlog dl m graph_base
    ascendcl ascendc_runtime)

target_link_directories(sparse_attn_ops PRIVATE
    ${ACL_LIB_DIR}
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    ${TORCH_NPU_LIBRARIES})

target_compile_options(sparse_attn_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)

message(STATUS "sparse_attn CMake configured")
