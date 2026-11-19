cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_C_COMPILER "tecocc")
set(CMAKE_CXX_COMPILER "tecocc")

project(teco_ops)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(TECOOPS_SOURCE_ROOT "${OPS_DIR}/.." ABSOLUTE)
# SDAA device LTO incorporates source paths into the fatbin layout.
# Normalize checkout paths so identical sources produce identical binaries.
set(TECOOPS_CANONICAL_SOURCE_ROOT "/root/teco-ops-tcdd-20260827-v229-v202-packed-compact-reduce")
set(REPRODUCIBLE_PATH_OPTIONS
    "-ffile-prefix-map=${TECOOPS_SOURCE_ROOT}=${TECOOPS_CANONICAL_SOURCE_ROOT}"
    "-fdebug-prefix-map=${TECOOPS_SOURCE_ROOT}=${TECOOPS_CANONICAL_SOURCE_ROOT}"
    "-fmacro-prefix-map=${TECOOPS_SOURCE_ROOT}=${TECOOPS_CANONICAL_SOURCE_ROOT}"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Werror)

option(WITH_TORCH "Enable PyTorch support" OFF)
if(WITH_TORCH)
    find_package(Python3 REQUIRED COMPONENTS Development)
    find_package(Torch REQUIRED)
    add_definitions(-DWITH_TORCH)
    include_directories(${TORCH_INCLUDE_DIRS})
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${TORCH_SDAA_HOME}/include)
    get_filename_component(TORCH_LIB_DIR ${TORCH_LIBRARY} DIRECTORY)
    message(STATUS "WITH_TORCH is enabled")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "TORCH_LIB_DIR: ${TORCH_LIB_DIR}")
else()
    message(STATUS "WITH_TORCH is disabled")
endif()

option(WITH_INFERENCE_PLUGIN "Enable Teco-inference Plugin support" OFF)
if(WITH_INFERENCE_PLUGIN)
    add_definitions(-DWITH_INFERENCE_PLUGIN)
    file(GLOB_RECURSE SRC_PLUGIN "${OPS_DIR}/plugin/*/*.cc")
    set(FIND_SDAART "/opt/tecoai/lib64")
    set(FIND_PluginUtil "$ENV{TECO_INFER_PLUGIN_UTIL_PATH}")
    set(PLUGIN_LIBS ${FIND_SDAART}/libsdaart.so ${FIND_PluginUtil}/libTecoInferPluginUtil.so)
    set(PLUGIN_INCLUDES
        -I/opt/tecoai/include
        -I/opt/tecoai/include/dyn/include
        -I/opt/tecoai/include/dyn/src
        -I/opt/tecoai/include/teco_infer_dyn
        -I${OPS_DIR}
    )
    set(PLUGIN_COMPILE_FLAGS -O3 -std=c++17 -fPIC -DWITH_INFERENCE_PLUGIN)
    set(PLUGIN_OBJ_DIR ${CMAKE_BINARY_DIR}/plugin_objs)
    set(PLUGIN_OBJECTS "")
    add_custom_target(plugin_mkdir ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${PLUGIN_OBJ_DIR})
    foreach(src ${SRC_PLUGIN})
        get_filename_component(src_name ${src} NAME)
        list(APPEND PLUGIN_OBJECTS ${PLUGIN_OBJ_DIR}/${src_name}.o)
        add_custom_command(OUTPUT ${PLUGIN_OBJ_DIR}/${src_name}.o
            COMMAND g++ -c ${src} -o ${PLUGIN_OBJ_DIR}/${src_name}.o ${PLUGIN_COMPILE_FLAGS} ${PLUGIN_INCLUDES}
            DEPENDS ${src}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endforeach()
    add_custom_target(plugin_compile ALL DEPENDS plugin_mkdir ${PLUGIN_OBJECTS})
    message(STATUS "WITH_INFERENCE_PLUGIN is enabled")
else()
    message(STATUS "WITH_INFERENCE_PLUGIN is disabled")
endif()

set(TECOOPS_OPS_PATH ${OPS_DIR}/interface/ops)

file(GLOB_RECURSE SRC_INTERFACE_COMMON "${OPS_DIR}/interface/common/*.cpp")
file(GLOB_RECURSE SRC_INTERFACE_OPS "${OPS_DIR}/interface/ops/*.cpp")
file(GLOB_RECURSE SRC_UAL_COMMON "${OPS_DIR}/ual/com/*.cpp")
file(GLOB_RECURSE SRC_UAL_OPS "${OPS_DIR}/ual/ops/*/*.cpp")
file(GLOB_RECURSE SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/*/*.scpp")

list(APPEND SRC_ALL
    ${SRC_INTERFACE_COMMON}
    ${SRC_INTERFACE_OPS}
    ${SRC_UAL_COMMON}
    ${SRC_UAL_OPS}
    ${SRC_UAL_KERNEL})

set(complie_options -O3 -msimd -mslave -fPIC -flto -x sdaa -std=c++17 --sdaa-device-lib=${OPS_DIR}/../thirdparty/teco-hal/lib/teco_hal.bc)
add_definitions(-DUSE_TECO)

add_library(ops_objs OBJECT ${SRC_ALL})
set_source_files_properties(${SRC_UAL_KERNEL} PROPERTIES LANGUAGE CXX)
set_target_properties(ops_objs PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(ops_objs PRIVATE ${complie_options} ${REPRODUCIBLE_PATH_OPTIONS})
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../common")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../thirdparty/teco-hal/include")

if(WITH_TORCH)
    set(TORCH_LINK_FLAGS "-L${TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10")
else()
    set(TORCH_LINK_FLAGS "")
endif()

if(WITH_INFERENCE_PLUGIN)
    set(TECO_OPS_PLUGIN_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops_plugin.so)
    set(TECO_INFER_PLUGIN_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libTecoInferPlugin.so)

    add_custom_target(ops_lib ALL
            COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND tecocc $<TARGET_OBJECTS:ops_objs>
            -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
            -o ${TECO_OPS_PLUGIN_LIB}
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(ops_lib ops_objs)

    add_custom_target(plugin_lib ALL
            COMMAND gcc ${PLUGIN_OBJECTS}
            -L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lteco_ops_plugin
            ${PLUGIN_LIBS}
            -shared -o ${TECO_INFER_PLUGIN_LIB}
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(plugin_lib ops_lib plugin_compile)
else()
    add_custom_target(ops_lib ALL
            COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND tecocc $<TARGET_OBJECTS:ops_objs> -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
            ${TORCH_LINK_FLAGS}
            -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops.so
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(ops_lib ops_objs)
endif()

set(CMAKE_CXX_LINK_FLAGS "-L$ENV{SDAA_HOME}/lib64  -lsdaart")