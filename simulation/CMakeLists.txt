cmake_minimum_required(VERSION 3.20)

project(eels_simulation VERSION 0.1.0 LANGUAGES CXX)

option(EELS_SIM_BUILD_GEANT4 "Build the Geant4 detector transport stage" ON)

if(EELS_SIM_BUILD_GEANT4)
  add_subdirectory(cpp)
endif()
