# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(stem_daqiri LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA architectures: GB10 (Spark) is sm_121 (compute capability 12.1).
# Include common lab archs so binaries are portable across host machines.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89;90;120;121")
endif()

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Capability opt-in flags. These keep the hello/link-check, TX, and RX build
# surfaces independently testable:
#
#   Hello/link check: both OFF (default)
#   TX-only:          STEM_DAQIRI_BUILD_TX=ON
#   TX+RX:            STEM_DAQIRI_BUILD_TX=ON and STEM_DAQIRI_BUILD_RX=ON
#
# The common CUDA kernels + pacing helpers are only built when at least one
# of TX/RX is enabled. The default only builds the hello/link-check binary.
# ---------------------------------------------------------------------------
option(STEM_DAQIRI_BUILD_TX "Build the STEM daqiri TX binary" OFF)
option(STEM_DAQIRI_BUILD_RX "Build the STEM daqiri RX binary" OFF)
option(STEM_DAQIRI_ENABLE_HDF5 "Enable the RX debug HDF5 writer when HDF5 is available" ON)
option(STEM_DAQIRI_REQUIRE_HDF5 "Require HDF5 support for parity-capable RX/replay builds" OFF)

if(STEM_DAQIRI_REQUIRE_HDF5)
  set(STEM_DAQIRI_ENABLE_HDF5 ON CACHE BOOL "Enable HDF5 support" FORCE)
endif()
if(STEM_DAQIRI_ENABLE_HDF5)
  if(STEM_DAQIRI_REQUIRE_HDF5)
    find_package(HDF5 COMPONENTS CXX REQUIRED)
  else()
    find_package(HDF5 COMPONENTS CXX QUIET)
  endif()
  if(HDF5_FOUND)
    message(STATUS "HDF5 found; stem_daqiri_rx will include the debug HDF5 writer")
  else()
    message(WARNING "HDF5 not found; stem_daqiri_rx will build with writer.noop support only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Bring daqiri in as a subproject. We disable daqiri's own examples to avoid
# pulling in raw_bench_common (which we don't need) and force shared libs so
# the daqiri runtime can be loaded from /opt/daqiri/lib in the runtime image.
# ---------------------------------------------------------------------------
set(DAQIRI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DAQIRI_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
if(DEFINED DAQIRI_MGR AND NOT DEFINED DAQIRI_ENGINE)
  # Older daqiri used DAQIRI_MGR; current daqiri uses DAQIRI_ENGINE.
  set(DAQIRI_ENGINE "${DAQIRI_MGR}" CACHE STRING "daqiri engine backends" FORCE)
endif()
if(NOT DEFINED DAQIRI_ENGINE)
  set(DAQIRI_ENGINE "dpdk" CACHE STRING "daqiri engine backends" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/daqiri
                 ${CMAKE_CURRENT_BINARY_DIR}/_daqiri_build)

# yaml-cpp is brought in transitively by daqiri's add_subdirectory call.
if(TARGET yaml-cpp::yaml-cpp)
  set(STEM_DAQIRI_YAML_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(STEM_DAQIRI_YAML_TARGET yaml-cpp)
else()
  find_package(yaml-cpp REQUIRED)
  if(TARGET yaml-cpp::yaml-cpp)
    set(STEM_DAQIRI_YAML_TARGET yaml-cpp::yaml-cpp)
  else()
    set(STEM_DAQIRI_YAML_TARGET yaml-cpp)
  endif()
endif()

# ---------------------------------------------------------------------------
# Hello/link-check binary: links against daqiri, calls daqiri_init with the
# provided config (or skips it in --self-test mode), then shutdown. Used for
# build/link/run validation. Always built.
# ---------------------------------------------------------------------------
add_executable(stem_daqiri_hello common/stem_daqiri_hello.cpp)
target_include_directories(stem_daqiri_hello PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(stem_daqiri_hello PRIVATE
  daqiri::daqiri
  ${STEM_DAQIRI_YAML_TARGET}
)

# ---------------------------------------------------------------------------
# Shared CUDA kernels + pacing helpers, only compiled when TX or RX is on.
# ---------------------------------------------------------------------------
set(STEM_DAQIRI_HAVE_COMMON_LIB FALSE)
if(STEM_DAQIRI_BUILD_TX OR STEM_DAQIRI_BUILD_RX)
  add_library(stem_daqiri_common STATIC
    common/stem_kernels.cu
    common/stem_pacing.cpp
  )
  target_include_directories(stem_daqiri_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
  )
  target_link_libraries(stem_daqiri_common PUBLIC
    CUDA::cudart
  )
  set_target_properties(stem_daqiri_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  set(STEM_DAQIRI_HAVE_COMMON_LIB TRUE)
endif()

# ---------------------------------------------------------------------------
# TX binary
# ---------------------------------------------------------------------------
if(STEM_DAQIRI_BUILD_TX)
  add_executable(stem_daqiri_tx tx/stem_tx_main.cpp)
  target_link_libraries(stem_daqiri_tx PRIVATE
    stem_daqiri_common
    daqiri::daqiri
    CUDA::cudart
    ${STEM_DAQIRI_YAML_TARGET}
  )
endif()

# ---------------------------------------------------------------------------
# RX binary
# ---------------------------------------------------------------------------
if(STEM_DAQIRI_BUILD_RX)
  add_executable(stem_daqiri_rx rx/stem_rx_main.cpp)
  target_link_libraries(stem_daqiri_rx PRIVATE
    stem_daqiri_common
    daqiri::daqiri
    CUDA::cudart
    ${STEM_DAQIRI_YAML_TARGET}
  )
  if(STEM_DAQIRI_ENABLE_HDF5 AND HDF5_FOUND)
    target_compile_definitions(stem_daqiri_rx PRIVATE STEM_DAQIRI_HAVE_HDF5=1)
    target_include_directories(stem_daqiri_rx PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(stem_daqiri_rx PRIVATE ${HDF5_CXX_LIBRARIES})
  endif()
endif()

# ---------------------------------------------------------------------------
# Copy YAML configs next to the binaries.
# ---------------------------------------------------------------------------
set(STEM_DAQIRI_CONFIGS
  configs/hello_self_test.yaml
)
if(STEM_DAQIRI_BUILD_TX)
  list(APPEND STEM_DAQIRI_CONFIGS
    configs/stem_tx_spark.yaml
    configs/stem_tx_igx_loopback.yaml
  )
endif()
if(STEM_DAQIRI_BUILD_RX)
  list(APPEND STEM_DAQIRI_CONFIGS
    configs/stem_rx_spark.yaml
    configs/stem_rx_spark_hds.yaml
    configs/stem_rx_igx_production.yaml
    configs/stem_rx_igx_loopback.yaml
    configs/stem_rx_igx_loopback_hds.yaml
    configs/stem_replay_hdf5.yaml
  )
endif()

foreach(cfg IN LISTS STEM_DAQIRI_CONFIGS)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${cfg})
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/${cfg}
      ${CMAKE_CURRENT_BINARY_DIR}/${cfg}
      COPYONLY
    )
  endif()
endforeach()

# Install binaries + configs.
set(STEM_DAQIRI_INSTALL_TARGETS stem_daqiri_hello)
if(STEM_DAQIRI_BUILD_TX)
  list(APPEND STEM_DAQIRI_INSTALL_TARGETS stem_daqiri_tx)
endif()
if(STEM_DAQIRI_BUILD_RX)
  list(APPEND STEM_DAQIRI_INSTALL_TARGETS stem_daqiri_rx)
endif()

install(TARGETS ${STEM_DAQIRI_INSTALL_TARGETS}
  RUNTIME DESTINATION bin
)
foreach(cfg IN LISTS STEM_DAQIRI_CONFIGS)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${cfg})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${cfg} DESTINATION bin/configs)
  endif()
endforeach()
