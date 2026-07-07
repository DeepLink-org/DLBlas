# ============================================================================
# CMake 构建配置 - MTPBlock Ascend C 算子
# Target: Ascend910B2, DAV_2201, CANN 9.0.0
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(MTPBlock LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")
set(ASC_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include")

# ============================================================================
# Target 1: 可执行文件（直调验证）- ASC only, NO CXX mixing
# ============================================================================

add_executable(mtpblock_custom op_host/mtpblock_host.asc)

target_link_libraries(mtpblock_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(mtpblock_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(mtpblock_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ============================================================================
# Target 2: TORCH_LIBRARY 模块 (libmtpblock_ops.so) - CXX only, no ASC
# ============================================================================
# Note: MatmulApiTiling is used by the torch extension (C++) for TCubeTiling
# computation. The ASC host (Target 1) uses a simplified manual TCubeTiling.

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(mtpblock_ops SHARED
    op_kernel/k1_embed_fuse_kernel.asc
    op_kernel/k2_hc_pre_kernel.asc
    op_kernel/k3_attn_block_kernel.asc
    op_kernel/k4_hc_post_kernel.asc
    op_kernel/k5_moe_block_kernel.asc
    op_kernel/k6_mtp_head_kernel.asc
    op_host/matmul_tiling_helper.cpp
    op_extension/mtpblock_torch.cpp
    op_extension/register.cpp
)

target_link_libraries(mtpblock_ops PRIVATE
    torch_npu
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
    c_sec
)

target_link_directories(mtpblock_ops PRIVATE
    ${ACL_LIB_DIR}
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
    ${TORCH_NPU_LIBRARIES}
)

target_include_directories(mtpblock_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
    ${ASC_INCLUDE_DIR}
)

target_compile_options(mtpblock_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

message(STATUS "TORCH_LIBRARY module 'libmtpblock_ops.so' will be built")
