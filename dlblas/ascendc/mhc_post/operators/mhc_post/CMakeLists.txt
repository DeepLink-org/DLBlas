# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(mhc_post LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# Python paths for torch extension
execute_process(
    COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python3 -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ============================================================================
# Target 1: 可执行文件（直调验证）
# ============================================================================
add_executable(mhc_post
    op_host/mhc_post.asc
    op_kernel/mhc_post_kernel.asc
)

target_link_libraries(mhc_post PRIVATE
    tiling_api register platform unified_dlog dl m graph_base
)

target_include_directories(mhc_post PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(mhc_post PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# ============================================================================
# Target 2: PyTorch 扩展共享库（libadd_custom_ops.so）
# ============================================================================
if(EXISTS "${TORCH_DIR}/include" AND EXISTS "${TORCH_NPU_DIR}/include")
    add_library(mhc_post_ops SHARED
        op_kernel/mhc_post_kernel.asc
        op_extension/mhc_post_torch.cpp
        op_extension/register.cpp
    )

    target_include_directories(mhc_post_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
        ${ACL_INCLUDE_DIR}
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${TORCH_NPU_DIR}/include/third_party/acl/inc
    )

    target_link_directories(mhc_post_ops PRIVATE
        ${ACL_LIB_DIR}
        "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    )

    target_link_libraries(mhc_post_ops PRIVATE
        tiling_api register platform unified_dlog dl m graph_base
        ascendcl ascendc_runtime
        ${TORCH_DIR}/lib/libtorch.so
        ${TORCH_DIR}/lib/libtorch_cpu.so
        ${TORCH_NPU_DIR}/lib/libtorch_npu.so
    )

    target_compile_options(mhc_post_ops PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    )
endif()
