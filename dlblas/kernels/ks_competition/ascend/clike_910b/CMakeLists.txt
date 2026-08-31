cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(dlblas_ks_ascendc_kernels LANGUAGES ASC CXX)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_QUERY_ERROR
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_QUERY_STATUS
)
if(NOT TORCH_QUERY_STATUS EQUAL 0)
    message(FATAL_ERROR
        "${Python3_EXECUTABLE} cannot import torch: ${TORCH_QUERY_ERROR}"
    )
endif()
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PATH}/Torch)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_NPU_QUERY_ERROR
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_QUERY_STATUS
)
if(NOT TORCH_NPU_QUERY_STATUS EQUAL 0)
    message(FATAL_ERROR
        "${Python3_EXECUTABLE} cannot import torch_npu: ${TORCH_NPU_QUERY_ERROR}"
    )
endif()

add_library(dlblas_ks_ascendc_ops SHARED
    csrc/sparse_attention.asc
    csrc/indexer.asc
    csrc/sinkhorn.asc
)

target_include_directories(dlblas_ks_ascendc_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
)

target_link_directories(dlblas_ks_ascendc_ops PRIVATE ${TORCH_NPU_PATH}/lib)
target_link_libraries(dlblas_ks_ascendc_ops PRIVATE torch_npu)

target_compile_definitions(dlblas_ks_ascendc_ops PRIVATE
    __GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(dlblas_ks_ascendc_ops PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:-O3>
)
