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
