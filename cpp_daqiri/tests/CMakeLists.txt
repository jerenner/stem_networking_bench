cmake_minimum_required(VERSION 3.18)
project(stem_synthetic_host_tests LANGUAGES CXX)

# Standalone host-only tests: no CUDA, DAQIRI, yaml-cpp, or network required.
enable_testing()
add_executable(test_synthetic test_synthetic.cpp)
target_compile_features(test_synthetic PRIVATE cxx_std_17)
target_include_directories(test_synthetic PRIVATE ../common)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(test_synthetic PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME synthetic_layout_and_pacing COMMAND test_synthetic)

add_executable(test_fpga_tiles test_fpga_tiles.cpp)
target_compile_features(test_fpga_tiles PRIVATE cxx_std_17)
target_include_directories(test_fpga_tiles PRIVATE ../common)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(test_fpga_tiles PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME fpga_tile_headers_and_mapping COMMAND test_fpga_tiles)

add_executable(test_samples test_samples.cpp)
target_compile_features(test_samples PRIVATE cxx_std_17)
target_include_directories(test_samples PRIVATE ../common)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(test_samples PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME adc_sample_decoding COMMAND test_samples)

add_executable(test_frame_rows test_frame_rows.cpp)
target_compile_features(test_frame_rows PRIVATE cxx_std_17)
target_include_directories(test_frame_rows PRIVATE ../common)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(test_frame_rows PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME detector_row_orientation COMMAND test_frame_rows)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_test(NAME signed_kernel_math_cpu_emulation
    COMMAND ${CMAKE_COMMAND} -E env "CXX=${CMAKE_CXX_COMPILER}"
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_sample_kernels.py)
endif()
