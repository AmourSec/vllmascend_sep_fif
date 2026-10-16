# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(a5_shared_host_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Architecture-specific CANN directory containing include and lib64")
if(NOT EXISTS "${CANN_ROOT}/include/acl/acl_rt.h")
    message(FATAL_ERROR "Set CANN_ROOT to the CANN directory containing include/acl/acl_rt.h")
endif()
find_program(BISHENG NAMES bisheng HINTS "${CANN_ROOT}/bin" REQUIRED)
find_library(ACL_RT NAMES acl_rt PATHS "${CANN_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDC_RUNTIME NAMES ascendc_runtime PATHS "${CANN_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
set(KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/probe_kernel.o")
add_custom_command(OUTPUT "${KERNEL_OBJECT}"
    COMMAND "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=${CANN_ROOT}"
            "LD_LIBRARY_PATH=${CANN_ROOT}/lib64:$ENV{LD_LIBRARY_PATH}"
            "${BISHENG}" --npu-arch=dav-3510 -xasc -std=c++17 -O2 -fPIC
            --cce-auto-sync=off -mllvm -cce-vf-remove-membar=false
            "-I${CANN_ROOT}/include" "-I${CANN_ROOT}/asc/include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/probe_kernel.asc" -o "${KERNEL_OBJECT}"
    DEPENDS probe_kernel.asc VERBATIM)
set_source_files_properties("${KERNEL_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
add_library(a5_host_probe SHARED native.cpp "${KERNEL_OBJECT}")
target_include_directories(a5_host_probe PRIVATE "${CANN_ROOT}/include")
target_link_libraries(a5_host_probe PRIVATE "${ASCENDC_RUNTIME}" "${ACL_RT}" dl pthread)
set_target_properties(a5_host_probe PROPERTIES BUILD_RPATH "${CANN_ROOT}/lib64")
