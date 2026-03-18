# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(stem_networking_bench CXX CUDA)

# Dependencies
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Torch is found via CMAKE_PREFIX_PATH set in the Dockerfile
# (On x86 this points to LibTorch CXX11, on ARM this points to Pip Torch)
find_package(Torch REQUIRED PATHS "/home/jrenner/local/software/libtorch")

# FIX ABI MISMATCH caused by PyTorch defaulting to ABI=0
# Strip the flag from global flags
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Strip from TORCH_CXX_FLAGS
if(TORCH_CXX_FLAGS)
  string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# Strip from common Torch targets if they exist
foreach(t "torch" "Torch" "torch_cpu" "torch_cuda")
  if(TARGET ${t})
    get_target_property(opts ${t} INTERFACE_COMPILE_OPTIONS)
    if(opts)
      string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" opts "${opts}")
      set_property(TARGET ${t} PROPERTY INTERFACE_COMPILE_OPTIONS "${opts}")
    endif()
  endif()
endforeach()

# Force ABI=1
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

find_package(HDF5 COMPONENTS CXX HL REQUIRED)

include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.9.4
)
FetchContent_MakeAvailable(MatX)

# Add this line to define CUDA::cublas, CUDA::cufft, etc.
find_package(CUDAToolkit REQUIRED)

# Global variables
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

# Create the executable
add_executable(stem_networking_bench
  kernels.cu
  main.cpp
  hdf5_replayer_op.cpp
  stem_receiver_op.h
  pytorch_processor_op.cpp
  hdf5_writer_op.cpp
)
target_include_directories(stem_networking_bench 
  BEFORE
  PRIVATE
  /opt/nvidia/holoscan/include/3rdparty
  /opt/nvidia/holoscan/include
  ${TORCH_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(stem_networking_bench
  PRIVATE
  holoscan::core
  holoscan::advanced_network
  matx::matx
  ${TORCH_LIBRARIES}
  HDF5::HDF5
)

# Link against Holoscan's fmt library if bundled
find_library(FMT_LIBRARY NAMES fmt HINTS /opt/nvidia/holoscan/lib)
if(FMT_LIBRARY)
    target_link_libraries(stem_networking_bench PRIVATE ${FMT_LIBRARY})
endif()

# Configure manager support through compile definitions
if(NOT DEFINED ANO_MGR)
    # Initialize ANO_MGR with a default if not provided
    # Note: rivermax not added to the default list yet since it requires licensing
    set(ANO_MGR "dpdk" CACHE STRING "Manager type(s) list")
endif()
separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()

# Copy config file
add_custom_target(stem_networking_bench_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/run_with_hdf5.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/run_with_hdf5.yaml"
)
add_custom_target(stem_networking_bench_fpga_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/run_with_network_fpga.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/run_with_network_fpga.yaml"
)
add_dependencies(stem_networking_bench stem_networking_bench_yaml stem_networking_bench_fpga_yaml)

# Installation
install(TARGETS stem_networking_bench
        DESTINATION examples/stem_networking_bench
        COMPONENT stem_networking_bench-cpp)

install(
  FILES
    run_with_hdf5.yaml
  DESTINATION examples/stem_networking_bench
  COMPONENT stem_networking_bench-configs
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES CMakeLists.txt.install
  RENAME CMakeLists.txt
  DESTINATION examples/stem_networking_bench
  COMPONENT stem_networking_bench-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES
    main.cpp
    kernels.cu
    kernels.cuh
    hdf5_replayer_op.cpp
    hdf5_writer_op.cpp
    pytorch_bench_op_rx.h
    stem_bench_op_tx.h
  DESTINATION examples/stem_networking_bench
  COMPONENT stem_networking_bench-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

