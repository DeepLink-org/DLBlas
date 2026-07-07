# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Sinkhorn Normalize examples build configuration
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(ACLNN_EXAMPLE)

add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(test_aclnn_sinkhorn_normalize
    test_aclnn_sinkhorn_normalize.cpp
)

add_executable(test_precision
    test_precision.cpp
)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann-9.0.0")
endif()

# Find custom operator header
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_sinkhorn_normalize.h
    PATHS
        ${ASCEND_PATH}/opp/vendors/sinkhorn_normalize_custom/op_api/include
        /usr/local/Ascend/opp/vendors/sinkhorn_normalize_custom/op_api/include
)

if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "Custom op header aclnn_sinkhorn_normalize.h not found. Please install operator package first.")
endif()
message(STATUS "Custom op header dir: ${CUSTOM_OP_INCLUDE_DIR}")

# Find custom operator library
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${ASCEND_PATH}/opp/vendors/sinkhorn_normalize_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/sinkhorn_normalize_custom/op_api/lib
)

if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "Custom op library libcust_opapi.so not found. Please install operator package first.")
endif()

include_directories(
    ${ASCEND_PATH}/include
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(test_aclnn_sinkhorn_normalize PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi.so
    ${CUSTOM_OP_LIBRARY}
)
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
target_link_options(test_aclnn_sinkhorn_normalize PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
)

target_include_directories(test_precision PRIVATE
    ${ASCEND_PATH}/include
    ${CUSTOM_OP_INCLUDE_DIR}
)
target_link_libraries(test_precision PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi.so
    ${CUSTOM_OP_LIBRARY}
)
target_link_options(test_precision PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
)

install(TARGETS test_aclnn_sinkhorn_normalize DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
